"""Train the model"""

# https://tensorflow.google.cn/guide/estimators?hl=zh-cn

'''
如果采用预创建的 Estimator，则有人已为您编写了模型函数。
如果采用自定义 Estimator，则您必须自行编写模型函数。
'''

import argparse
import os

import tensorflow as tf

from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)

    '''
    model_dir: 保存模型参数、图等的地址，也可以用来将路径中的检查点加载至 estimator 中来继续训练之前保存的模型。如果是 PathLike， 那么路径就固定为它了。如果是 None，那么 config 中的 model_dir 会被使用（如果设置了的话），如果两个都设置了，那么必须相同；如果两个都是 None，则会使用临时目录。
    config: 配置类。
    params: 超参数的dict，会被传递到 model_fn。keys 是参数的名称，values 是基本 python 类型。
    
    def model_fn(features, labels, mode, params):
    '''
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(args.data_dir, params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
