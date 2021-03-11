#!/bin/bash
TRACKING_URI= ./logs/mlruns
mlflow ui --backend-store-uri $TRACKING_URI