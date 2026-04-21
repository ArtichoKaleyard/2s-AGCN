"""2s-AGCN 项目层面向 Foundry 的显式注册桥。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .data import build_legacy_split_datasets
from .models import AAGCNModel, AGCNModel, TwoStreamSkeletonModel
from .models.graph import get_graph_spec


def _read_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    """从 mapping 或简单配置对象中读取 ``name``。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dataset_spec_value(spec: Any, name: str, default: Any) -> Any:
    """读取 Foundry dataset spec 元数据，同时避免依赖具体类型。"""

    return _read_attr_or_key(spec, name, default)


def _resolve_graph_params(model_params: Mapping[str, Any], dataset_spec: Any) -> tuple[str, str]:
    """解析 Foundry 已经选定的低层 graph 参数。"""

    graph = model_params.get("graph", {}) or {}
    layout = (
        model_params.get("graph_layout")
        or _read_attr_or_key(graph, "layout")
        or _read_attr_or_key(graph, "layout_name")
        or _dataset_spec_value(dataset_spec, "layout_name", "ntu-rgb+d")
    )
    strategy = (
        model_params.get("graph_strategy")
        or _read_attr_or_key(graph, "strategy")
        or "spatial"
    )
    return str(layout), str(strategy)


def _resolve_model_shape(model_params: Mapping[str, Any], dataset_spec: Any) -> dict[str, int]:
    """从 Foundry 注入的 dataset 元数据中解析模型尺寸。"""

    return {
        "num_class": int(
            model_params.get(
                "num_class",
                model_params.get("num_classes", _dataset_spec_value(dataset_spec, "num_classes", 60)),
            )
        ),
        "num_point": int(
            model_params.get(
                "num_point",
                model_params.get("num_joints", _dataset_spec_value(dataset_spec, "num_joints", 25)),
            )
        ),
        "num_person": int(
            model_params.get(
                "num_person",
                model_params.get("num_persons", _dataset_spec_value(dataset_spec, "num_persons", 2)),
            )
        ),
        "in_channels": int(model_params.get("in_channels", _dataset_spec_value(dataset_spec, "in_channels", 3))),
    }


def _build_single_stream(
    model_cls: type[AGCNModel] | type[AAGCNModel],
    model_params: Mapping[str, Any],
) -> nn.Module:
    """根据具体低层参数构建一个 AGCN 系列 stream。"""

    dataset_spec = model_params.get("_dataset_spec")
    layout, strategy = _resolve_graph_params(model_params, dataset_spec)
    shape = _resolve_model_shape(model_params, dataset_spec)
    kwargs: dict[str, Any] = {
        **shape,
        "graph_layout": layout,
        "graph_strategy": strategy,
    }
    if model_cls is AAGCNModel:
        kwargs["drop_out"] = float(model_params.get("drop_out", 0))
        kwargs["adaptive"] = bool(model_params.get("adaptive", True))
        kwargs["attention"] = bool(model_params.get("attention", True))
    return model_cls(**kwargs)


def _normalize_streams(model_params: Mapping[str, Any]) -> tuple[str, ...]:
    """读取 Foundry 传入的 stream 名称，不重新校验 stream 语义。"""

    streams = model_params.get("streams")
    stream_mode = str(model_params.get("stream_mode", "joint"))
    if streams is None:
        return ("joint", "bone") if stream_mode == "two_stream" else (stream_mode,)
    if isinstance(streams, str):
        return (streams,)
    return tuple(str(stream) for stream in streams)


def _build_model(model_cls: type[AGCNModel] | type[AAGCNModel], model_params: Mapping[str, Any]) -> nn.Module:
    """根据 Foundry model params 构建单流或双流模型。"""

    stream_mode = str(model_params.get("stream_mode", "joint"))
    streams = _normalize_streams(model_params)
    if stream_mode == "two_stream" or len(streams) > 1:
        shape = _resolve_model_shape(model_params, model_params.get("_dataset_spec"))
        stream_builders = {
            stream: (lambda cls=model_cls, params=dict(model_params): _build_single_stream(cls, params))
            for stream in streams
        }
        return TwoStreamSkeletonModel(
            stream_builders,
            stream_mode="two_stream",
            fusion=model_params.get("fusion", "sum"),
            num_classes=shape["num_class"],
        )
    return _build_single_stream(model_cls, model_params)


def build_agcn_model(model_params: Mapping[str, Any], device: Any | None = None) -> nn.Module:
    """官方 AGCN 实现的 Foundry model builder。"""

    model = _build_model(AGCNModel, model_params)
    return model.to(device) if device is not None else model


def build_aagcn_model(model_params: Mapping[str, Any], device: Any | None = None) -> nn.Module:
    """官方 AAGCN 实现的 Foundry model builder。"""

    model = _build_model(AAGCNModel, model_params)
    return model.to(device) if device is not None else model


def _dataset_spec_kwargs(name: str, layout: str, num_classes: int, num_joints: int) -> dict[str, Any]:
    """根据项目侧旧数据适配事实构造 dataset spec 参数。"""

    graph = get_graph_spec(layout)
    return {
        "num_classes": num_classes,
        "num_joints": num_joints,
        "num_persons": 2,
        "in_channels": 3,
        "layout_name": graph.layout,
        "bones": graph.inward,
        "metadata": {"legacy_adapter": "two_stream_agcn", "dataset_name": name},
    }


def _project_dataset_spec(name: str, layout: str, num_classes: int, num_joints: int) -> Any:
    """复用 Foundry 已注册 dataset spec，仅在缺失时退回项目侧最小事实。"""

    from foundry import DatasetSpec, RegistryError
    from foundry.core.registry import get_dataset_spec

    try:
        return get_dataset_spec(name)
    except RegistryError:
        return DatasetSpec(**_dataset_spec_kwargs(name, layout, num_classes, num_joints))


def register_two_stream_agcn_project() -> None:
    """向 Foundry 注册项目侧 builders。

    该函数刻意只注册具体模型与旧数据适配构建器，不注册 skeleton 编译器、
    protocol 规则、stream/fusion 规则、graph 校验、dataset alias 映射或高层配置编译。
    """

    import foundry.projects.skeleton  # noqa: F401 - 确保 Foundry 先注册 skeleton dataset spec。
    from foundry import register_dataset, register_model

    register_model("two_stream_agcn.agcn", build_agcn_model)
    register_model("two_stream_agcn.aagcn", build_aagcn_model)

    register_dataset(
        "ntu_rgbd60",
        build_legacy_split_datasets,
        _project_dataset_spec("ntu_rgbd60", "ntu-rgb+d", 60, 25),
    )
    register_dataset(
        "kinetics_skeleton",
        build_legacy_split_datasets,
        _project_dataset_spec("kinetics_skeleton", "openpose18", 400, 18),
    )
