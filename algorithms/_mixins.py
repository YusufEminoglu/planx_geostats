# -*- coding: utf-8 -*-
"""Shared helpUrl mixin for all GeoStats algorithms."""

DOC_BASE_URL = "https://yusufeminoglu.github.io/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html"

class HelpUrlMixin:
    def helpUrl(self) -> str:
        return DOC_BASE_URL + "#alg-" + self.name()
