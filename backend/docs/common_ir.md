# Common IR

This document describes the v1 common schema and performance IR used to normalize profiler and telemetry sources into one canonical representation.

## Purpose

The IR exists so downstream analysis, comparison, and recommendation logic can operate on one language instead of source-specific formats.

## V1 entities

* `Run`
* `Event`
* `Metric`
* `Annotation`

## V1 event families

* `kernel`
* `memory`
* `cpu`
* `data_pipeline`
* `distributed`
* `annotation`

## V1 metric family

* `metric`

## V1 rules

* preserve raw source payloads separately from canonical IR records
* keep normalized identifiers and timestamps canonical
* keep source-specific fields in `attrs`
