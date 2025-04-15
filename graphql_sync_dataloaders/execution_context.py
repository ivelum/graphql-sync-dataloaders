from typing import (
    Any,
    Dict,
    Optional,
    List,
    Iterable,
    Union,
)
from functools import partial

from graphql import (
    ExecutionContext,
    FieldNode,
    GraphQLError,
    GraphQLObjectType,
    GraphQLOutputType,
    GraphQLResolveInfo,
    GraphQLList,
    OperationDefinitionNode,
    located_error,
)
from graphql.pyutils import (
    is_iterable,
    Path,
    AwaitableOrValue,
    Undefined,
)

from graphql.execution.execute import get_field_def
from graphql.execution.values import get_argument_values

from .sync_future import SyncFuture, maybe_then
from .sync_dataloader import dataloader_batch_callbacks


class PendingFuture:
    def __str__(self):
        return 'PENDING_FUTURE'


PENDING_FUTURE = PendingFuture()


class DeferredExecutionContext(ExecutionContext):
    """Execution for working with synchronous Futures.

    This execution context can handle synchronous Futures as resolved values.
    Deferred callbacks set in these Futures are called after the operation
    is executed and before the result is returned.
    """

    def execute_operation(
        self, operation: OperationDefinitionNode, root_value: Any
    ) -> Optional[AwaitableOrValue[Any]]:
        result = super().execute_operation(operation, root_value)

        dataloader_batch_callbacks.run_all_callbacks()

        if isinstance(result, SyncFuture):
            if not result.done():
                raise RuntimeError("GraphQL deferred execution failed to complete.")
            return result.result()

        return result

    def execute_fields_serially(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Optional[Path],
        fields: Dict[str, List[FieldNode]],
    ) -> Union[AwaitableOrValue[Dict[str, Any]], SyncFuture]:
        results: AwaitableOrValue[Dict[str, Any]] = {}

        future = SyncFuture()

        unresolved = 0
        callbacks = []
        for response_name, field_nodes in fields.items():
            field_path = Path(path, response_name, parent_type.name)

            # Add placeholder so that field order is preserved
            results[response_name] = PENDING_FUTURE
            unresolved += 1

            # noinspection PyShadowingNames
            def process_result(response_name, result):
                nonlocal unresolved

                if result is not Undefined:
                    results[response_name] = result
                else:
                    del results[response_name]

                unresolved -= 1
                if not unresolved:
                    future.set_result(results)

            callbacks.append((
                self.execute_field(parent_type, source_value, field_nodes, field_path),
                partial(process_result, response_name),
            ))

        for field_result, callback in callbacks:
            maybe_then(field_result, callback)

        if future.done():
            return future.result()
        return future

    execute_fields = execute_fields_serially

    def execute_field(
        self,
        parent_type: GraphQLObjectType,
        source: Any,
        field_nodes: List[FieldNode],
        path: Path,
    ) -> AwaitableOrValue[Any]:
        field_def = get_field_def(self.schema, parent_type, field_nodes[0])
        if not field_def:
            return Undefined
        return_type = field_def.type
        resolve_fn = field_def.resolve or self.field_resolver
        if self.middleware_manager:
            resolve_fn = self.middleware_manager.get_field_resolver(resolve_fn)
        info = self.build_resolve_info(field_def, field_nodes, parent_type, path)
        args = get_argument_values(field_def, field_nodes[0], self.variable_values)

        def on_exception(e, future=None):
            located = located_error(e, field_nodes, path.as_list())
            self.handle_field_error(located, return_type)
            if future is not None:
                future.set_result(None)

        try:
            result = resolve_fn(source, info, **args)
        except Exception as e:
            on_exception(e)
            return None

        complete_value = partial(self.complete_value, return_type, field_nodes, info, path)
        return maybe_then(result, complete_value, on_exception)

    def complete_list_value(
        self,
        return_type: GraphQLList[GraphQLOutputType],
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: SyncFuture | Iterable[Any],
    ):
        callback = partial(self._complete_list_value, return_type, field_nodes, info, path)
        return maybe_then(result, callback)

    def _complete_list_value(
        self,
        return_type: GraphQLList[GraphQLOutputType],
        field_nodes: List[FieldNode],
        info: GraphQLResolveInfo,
        path: Path,
        result: Iterable[Any],
    ) -> Union[AwaitableOrValue[List[Any]], SyncFuture]:
        if not is_iterable(result):
            raise GraphQLError(
                "Expected Iterable, but did not find one for field"
                f" '{info.parent_type.name}.{info.field_name}'."
            )

        future = SyncFuture()

        item_type = return_type.of_type
        results: List[Any] = [None] * len(result)

        unresolved = 0
        callbacks = []
        for index, item_or_future in enumerate(result):
            item_path = path.add_key(index, None)

            unresolved += 1

            # noinspection PyShadowingNames
            def process_item(index, item_path, item):
                nonlocal unresolved

                completed_value_or_future = self.complete_value(
                    item_type, field_nodes, info, item_path, item
                )

                def on_completed(completed_value):
                    nonlocal unresolved

                    results[index] = completed_value

                    unresolved -= 1
                    if not unresolved:
                        future.set_result(results)

                maybe_then(completed_value_or_future, on_completed)

            # noinspection PyShadowingNames
            def on_exception(item_path, e):
                nonlocal unresolved

                unresolved -= 1
                if not unresolved:
                    future.set_result(results)

                error = located_error(
                    e, field_nodes, item_path.as_list()
                )
                self.handle_field_error(error, item_type)

            callbacks.append((
                item_or_future,
                partial(process_item, index, item_path),
                partial(on_exception, item_path),
            ))

        for item_or_future, callback, on_exception in callbacks:
            maybe_then(item_or_future, callback, on_exception)

        if not unresolved:
            return results

        return future
