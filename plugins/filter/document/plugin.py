from common.plugin.helper import FilterPluginCore
from common.plugin.model import Meta
from typing import Any, List, Union
from logging import Logger

class Plugin(FilterPluginCore):
    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self._logger.info("FilterPluginCore init...")
        self.meta = Meta(
            name="Filter Plugin",
            description="Filter plugin template",
            version="0.0.1",
        )

    def invoke(self, config_data: dict):
        self._logger.info("FilterPluginCore invoke...")
        self._filters = config_data
        self._logger.debug("self.config - [%s]", self._filters)
        return self

    def _get_nested_value(self, tid:str, obj: Any, path: str) -> Any:
        self._logger.debug(f"TID: %s, object: %s, path: %s", tid, obj, path)
        keys = path.split(".")
        for key in keys:
            self._logger.debug("TID:%s, key: %s", tid, key)
            if not key:
                obj = None
                break
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                obj = None
                break
        self._logger.debug(f"TID: %s, _get_nested_value: %s", tid, obj)
        return obj

    def apply_filter(self, **kwargs: Any) -> List[Any]:
        tid = kwargs.get("tid", None)
        documents =  kwargs.get("documents", None)
        if self._filters is None:
            return documents #skip filter
        result = []
        for item in documents:
            filtered = False
            for filter_condition in self._filters:
                if self._apply_filter(
                    tid,
                    filter_condition["term"],
                    self._get_nested_value(tid, item, filter_condition["path"]),
                    filter_condition["value"],
                ):
                    filtered = True
                    break
            if filtered is False:
                result.append(item)

        return result
    def _apply_filter(self, tid:str, term: str,
        field_value: Any, filter_values: Union[Any, List[Any]]
    ) -> bool:
        self._logger.debug(f"TID: %s, term: %s, field_value: %s, filter_values: %s",
            tid, term, field_value, filter_values
        )
        if field_value is None:
            return False
        if term == "match":
            return field_value in filter_values
        elif term == "not_match":
            return field_value not in filter_values
        elif term == "greater_than":
            return field_value > filter_values
        elif term == "less_than":
            return field_value < filter_values
        else:
            return True
