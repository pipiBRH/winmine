- 使用預先訓練的 model
~~~bash
$> python gui.py -t ./model/readymade/model-gamma-3500.meta -m model-gamma.py
~~~

- 訓練 model
~~~bash
$> python model-gamma.py
~~~

- 查看訓練結果
~~~bash
$> python gui.py -t ./model/model-gamma/run-{date}-{time}/model-gamma-{check-point}.meta -m model-gamma.py
~~~

- 使用 TensorBoard 查看結果
~~~bash
$> tensorboard --logdir ./model/model-gamma/run-{date}-{time}/
~~~

