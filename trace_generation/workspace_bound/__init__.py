"""
workspace_bound 包 - 向后兼容层

新代码应该使用: data.workspace_bounds
"""

try:
    from data.workspace_bounds.workspace_analyzer import WorkspaceAnalyzer
except ImportError:
    from .workspace_analyzer import WorkspaceAnalyzer

__all__ = ['WorkspaceAnalyzer']
