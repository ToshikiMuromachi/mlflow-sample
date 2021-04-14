import mlflow

# runIDを発行する
mlflow.start_run()

# パラメータを記録する (key-value pair)
mlflow.log_param("param1", 5)

# メトリックを記録する(ステップごとに)
mlflow.log_metric("foo", 2, step=1)
mlflow.log_metric("foo", 4, step=2)
mlflow.log_metric("foo", 6, step=3)

# 生成物を記録する (output file)
with open("../data/output/output.txt", "w") as f:
    f.write("Hello world!")
mlflow.log_artifact("../data/output/output.txt")

mlflow.end_run()
