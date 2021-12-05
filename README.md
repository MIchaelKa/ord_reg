# Searching best ordinal regression method
This repo provides several experiments to compare different methods suitable for the ordinal regression task. To run any of those experiments use the command:
```
python main.py +experiment=experiment_name
```

For the full list of available experiments look at Methods section.

You can also change many other configs, for example: CNN encoder, learning rate, scheduler, optimizer, etc. For the full list of available configs see the base config file located under `config/base.yaml`. Some of the experiments already contain some predefined config customization.
For example:
```
python main.py +experiment=label_bin model.encoder.model_name=resnext50_32x4d optimizer=sgd optimizer.lr=3e-4
```


You can use tensorboard for monitoring the training process. Run this command and then access the web app at the following url: [http://localhost:6006/](http://localhost:6006/)
```
tensorboard --logdir=/path_to_project/ord_reg/ord_reg/outputs/runs
```

For all our experiments we use RetinaMNIST dataset
> The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity.

For more details visit: [MedMNIST github repo](https://github.com/MedMNIST/MedMNIST)

# Methods
## Baseline
The simplest way is to use cross-entropy loss and predict different grades as different classes. This approach doesn't do any attempts to encounter ordinal relations between grades and this can lead to some inconsistency between predictions.

Let's assume we have ordinal categories or grades: `A < B < C < D`

This method can predict high probabilities for A and C and low for B which is not appropriate.

We provide this as a baseline experiment to make a comparison with other approaches.

### Command
```
python main.py +experiment=baseline
```

## Label binning
This approach transforms ordinal regression problem into `N - 1` binary classification problems, where `N` is the initial number of grades/classes/ordinal categories.


We preprocess our data in the following way, where each binary label b_i indicates whether this sample exceeds the grade r_i.

Some examples:
```
Grade 0: label = [0,0,0,0]
Grade 2: label = [1,1,0,0]
Grade 4: label = [1,1,1,1]
```

Now the difference between categories follows our ordinal scale, so the difference between Grade 0 and Grade 1 is less than between Grade 0 and Grade 3. In the same way, it helps to penalize the model more when it makes larger mistakes on the ordinal scale during training.

Indeed this method works better in practice but still can suffer from inconsistency between predicted probabilities.

### References

1. A neural network approach to ordinal regression (https://arxiv.org/abs/0704.1028)
2. Ordinal Regression with Multiple Output CNN for Age Estimation (https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S21-20.pdf)

### Command
```
python main.py +experiment=label_bin
```

## COnsistent RAnk Logits (CORAL)
This method restricts the neural network to make rank-consistent predictions. 

To achieve rank-monotonicity and guarantee binary classifier consistency, the binary tasks share the same weight parameters but have independent bias units. We can also pre-initialize the biases to descending values in [0, 1] range as described in the paper *this pre-initialization scheme results in faster learning and better generalization performance in practice.*

The paper also provides theoretical proves for classifier consistency which was confirmed in our experiments.


### References
1. Rank consistent ordinal regression for neural networks with application to age estimation (https://arxiv.org/abs/1901.07884)

### Command
```
python main.py +experiment=coral
```

## TBD
### Regression approach

MSE/MAE

### Siamese CNN architecture
Compute the rank from pair-wise comparisons between the input image and multiple, carefully selected anchor images

Ordinal Regression using Noisy Pairwise Comparisons for Body Mass Index Range Estimation (https://arxiv.org/abs/1811.03268)

# Metric
We use **Quadratic Weighted Kappa** as a performance evaluation metric. This metric is better suitable for scoring ordinal predictions since it takes into account how much our prediction further away from the actual value.

Some links for better understanfing the metric

1. [understanding-the-quadratic-weighted-kappa](https://www.kaggle.com/reighns/understanding-the-quadratic-weighted-kappa)
2. [scikit-learn doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)

# Neural network architectures
We use timm library for trying various cnn encoders.

You can try any model supported by timm.
Example:
```
python main.py +experiment=label_bin model.model_name=resnext50_32x4d
```
For the full list of available architectures visit [timm github repo](https://github.com/rwightman/pytorch-image-models)

# Results
All the experiments were performed using 3 different random seeds and then averaging out the results. For the final evaluation, we use the test dataset provided within `RetinaMNIST` dataset.

The test dataset contains 400 more examples of retina fundus images.

Metric: Quadratic Weighted Kappa(QWK)

Inconsistency: The number of inconsistent predictions out from 400 total examples.

### resnet18
| Experiment  | QWK        | Inconsistency |
| ----------- | ---------- | --------------|
| baseline    | 0.4989 | 388/400 |
| label_bin   | **0.5623** | 148/400 |
| coral       | 0.543  | **0**/400 |

Despite 100% consistent predictions CORAL method works not so well on this dataset. At the same time, both methods based on label binning strategy outperforms the baseline.
