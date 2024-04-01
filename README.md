[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsorber/neo-ls-svm) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=739301655)

# Neo LS-SVM

Neo LS-SVM is a modern [Least-Squares Support Vector Machine](https://en.wikipedia.org/wiki/Least-squares_support_vector_machine) implementation in Python that offers several benefits over sklearn's classic `sklearn.svm.SVC` classifier and `sklearn.svm.SVR` regressor:

1. ⚡ Linear complexity in the number of training examples with [Orthogonal Random Features](https://arxiv.org/abs/1610.09072).
2. 🚀 Hyperparameter free: zero-cost optimization of the [regularisation parameter γ](https://en.wikipedia.org/wiki/Ridge_regression#Tikhonov_regularization) and [kernel parameter σ](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).
3. 🏔️ Adds a new tertiary objective that minimizes the complexity of the prediction surface.
4. 🎁 Returns the leave-one-out residuals and error for free after fitting.
5. 🌀 Learns an affine transformation of the feature matrix to optimally separate the target's bins.
6. 🪞 Can solve the LS-SVM both in the primal and dual space.
7. 🌡️ Isotonically calibrated `predict_proba`.
8. ✅ Conformally calibrated `predict_quantiles` and `predict_interval`.
9. 🔔 Bayesian estimation of the predictive standard deviation with `predict_std`.
10. 🐼 Pandas DataFrame output when the input is a pandas DataFrame.

## Using

### Installing

First, install this package with:

```bash
pip install neo-ls-svm
```

### Classification and regression

Then, you can import `neo_ls_svm.NeoLSSVM` as an sklearn-compatible binary classifier and regressor. Example usage:

```python
from neo_ls_svm import NeoLSSVM
from pandas import get_dummies
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Binary classification example:
X, y = fetch_openml("churn", version=3, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(get_dummies(X), y, test_size=0.15, random_state=42)
model = NeoLSSVM().fit(X_train, y_train)
model.score(X_test, y_test)  # 93.1% (compared to sklearn.svm.SVC's 89.6%)

# Regression example:
X, y = fetch_openml("ames_housing", version=1, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(get_dummies(X), y, test_size=0.15, random_state=42)
model = NeoLSSVM().fit(X_train, y_train)
model.score(X_test, y_test)  # 82.4% (compared to sklearn.svm.SVR's -11.8%)
```

### Predicting quantiles

Neo LS-SVM implements conformal prediction with a Bayesian nonconformity estimate to compute quantiles and prediction intervals for both classification and regression. Example usage:

```python
# Predict the house prices and their quantiles.
ŷ_test = model.predict(X_test)
ŷ_test_quantiles = model.predict_quantiles(X_test, quantiles=(0.025, 0.05, 0.1, 0.9, 0.95, 0.975))
```

When the input data is a pandas DataFrame, the output is also a pandas DataFrame. For example, printing the head of `ŷ_test_quantiles` yields:

|   house_id |    0.025 |     0.05 |      0.1 |      0.9 |     0.95 |    0.975 |
|-----------:|---------:|---------:|---------:|---------:|---------:|---------:|
|       1357 | 114283.0 | 124767.6 | 133314.0 | 203162.0 | 220407.5 | 245655.3 |
|       2367 |  85518.3 |  91787.2 |  93709.8 | 107464.3 | 108472.6 | 114482.3 |
|       2822 | 147165.9 | 157462.8 | 167193.1 | 243646.5 | 263324.4 | 291963.3 |
|       2126 |  81788.7 |  88738.1 |  91367.4 | 111944.9 | 114800.7 | 122874.5 |
|       1544 |  94507.1 | 108288.2 | 120184.3 | 222630.5 | 248668.2 | 283703.4 |

Let's visualize the predicted quantiles on the test set:

<img src="https://github.com/lsorber/neo-ls-svm/assets/4543654/cd24e739-e857-4045-8a70-07e92367a901" width="512">

<details>
<summary>Expand to see the code that generated the graph above</summary>

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

%config InlineBackend.figure_format = "retina"
plt.rcParams["font.size"] = 8
idx = (-ŷ_test.sample(50, random_state=42)).sort_values().index
y_ticks = list(range(1, len(idx) + 1))
plt.figure(figsize=(4, 5))
for j in range(3):
    end = ŷ_test_quantiles.shape[1] - 1 - j
    coverage = round(100 * (ŷ_test_quantiles.columns[end] - ŷ_test_quantiles.columns[j]))
    plt.barh(
        y_ticks,
        ŷ_test_quantiles.loc[idx].iloc[:, end] - ŷ_test_quantiles.loc[idx].iloc[:, j],
        left=ŷ_test_quantiles.loc[idx].iloc[:, j],
        label=f"{coverage}% Prediction interval",
        color=["#b3d9ff", "#86bfff", "#4da6ff"][j],
    )
plt.plot(y_test.loc[idx], y_ticks, "s", markersize=3, markerfacecolor="none", markeredgecolor="#e74c3c", label="Actual value")
plt.plot(ŷ_test.loc[idx], y_ticks, "s", color="blue", markersize=0.6, label="Predicted value")
plt.xlabel("House price")
plt.ylabel("Test house index")
plt.xlim(0, 500e3)
plt.yticks(y_ticks, y_ticks)
plt.tick_params(axis="y", labelsize=6)
plt.grid(axis="x", color="lightsteelblue", linestyle=":", linewidth=0.5)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend()
plt.tight_layout()
plt.show()
```
</details>

### Predicting intervals

In addition to quantile prediction, you can use `predict_interval` to predict conformally calibrated prediction intervals. Compared to quantiles, these focus on reliable coverage over quantile accuracy. Example usage:

```python
# Compute prediction intervals for the houses in the test set.
ŷ_test_interval = model.predict_interval(X_test, coverage=0.95)

# Measure the coverage of the prediction intervals on the test set
coverage = ((ŷ_test_interval.iloc[:, 0] <= y_test) & (y_test <= ŷ_test_interval.iloc[:, 1])).mean()
print(coverage)  # 94.3%
```

When the input data is a pandas DataFrame, the output is also a pandas DataFrame. For example, printing the head of `ŷ_test_interval` yields:

|   house_id |    0.025 |    0.975 |
|-----------:|---------:|---------:|
|       1357 | 114283.0 | 245849.2 |
|       2367 |  85518.3 | 114411.4 |
|       2822 | 147165.9 | 292179.2 |
|       2126 |  81788.7 | 122838.1 |
|       1544 |  94507.1 | 284062.6 |

## Benchmarks

We select all binary classification and regression datasets below 1M entries from the [AutoML Benchmark](https://arxiv.org/abs/2207.12560). Each dataset is split into 85% for training and 15% for testing. We apply `skrub.TableVectorizer` as a preprocessing step for `neo_ls_svm.NeoLSSVM` and `sklearn.svm.SVC,SVR` to vectorize the pandas DataFrame training data into a NumPy array. Models are fitted only once on each dataset, with their default settings and no hyperparameter tuning.

<details open>
<summary>Binary classification</summary>

ROC-AUC on 15% test set:

|                          dataset |   LGBMClassifier |         NeoLSSVM |              SVC |
|---------------------------------:|-----------------:|-----------------:|-----------------:|
|                              ada |  🥈 90.9% (0.1s) |  🥇 90.9% (1.9s) |     83.1% (4.5s) |
|                            adult |  🥇 93.0% (0.5s) | 🥈 89.0% (15.7s) |                / |
|           amazon_employee_access |  🥇 85.6% (0.5s) |  🥈 64.5% (9.0s) |                / |
|                           arcene |  🥈 78.0% (0.6s) |     70.0% (6.3s) |  🥇 82.0% (4.0s) |
|                       australian |  🥇 88.3% (0.2s) |     79.9% (1.7s) |  🥈 81.9% (0.1s) |
|                   bank-marketing |  🥇 93.5% (0.5s) | 🥈 91.0% (11.8s) |                / |
| blood-transfusion-service-center |     62.0% (0.3s) |  🥇 71.0% (2.2s) |  🥈 69.7% (0.1s) |
|                            churn |  🥇 91.7% (0.6s) |  🥈 81.0% (2.1s) |     70.6% (2.9s) |
|           click_prediction_small |  🥇 67.7% (0.5s) | 🥈 66.6% (10.9s) |                / |
|                          jasmine |  🥇 86.1% (0.3s) |     79.5% (1.9s) |  🥈 85.3% (7.4s) |
|                              kc1 |  🥇 78.9% (0.3s) |  🥈 76.6% (1.4s) |     45.7% (0.6s) |
|                         kr-vs-kp | 🥇 100.0% (0.6s) |     99.2% (1.6s) |  🥈 99.4% (2.3s) |
|                         madeline |  🥇 93.1% (0.5s) |     65.6% (1.9s) | 🥈 82.5% (19.8s) |
|                  ozone-level-8hr |  🥈 91.2% (0.4s) |  🥇 91.6% (1.7s) |     72.9% (0.6s) |
|                              pc4 |  🥇 95.3% (0.3s) |  🥈 90.9% (1.5s) |     25.7% (0.3s) |
|                 phishingwebsites |  🥇 99.5% (0.5s) |  🥈 98.9% (3.6s) |    98.7% (10.0s) |
|                          phoneme |  🥇 95.6% (0.3s) |  🥈 93.5% (2.1s) |     91.2% (2.0s) |
|                      qsar-biodeg |  🥇 92.7% (0.4s) |  🥈 91.1% (5.2s) |     86.8% (0.3s) |
|                        satellite |  🥈 98.7% (0.2s) |  🥇 99.5% (1.9s) |     98.5% (0.4s) |
|                          sylvine |  🥇 98.5% (0.2s) |  🥈 97.1% (2.0s) |     96.5% (3.8s) |
|                             wilt |  🥈 99.5% (0.2s) |  🥇 99.8% (1.8s) |     98.9% (0.5s) |

</details>

<details open>
<summary>Regression</summary>

R² on 15% test set:

|                       dataset |   LGBMRegressor |         NeoLSSVM |              SVR |
|------------------------------:|----------------:|-----------------:|-----------------:|
|                       abalone | 🥈 56.2% (0.1s) |  🥇 59.5% (2.5s) |     51.3% (0.7s) |
|                        boston | 🥇 91.7% (0.2s) |  🥈 89.6% (1.1s) |     35.1% (0.0s) |
|              brazilian_houses | 🥈 55.9% (0.3s) |  🥇 88.4% (3.7s) |      5.4% (7.0s) |
|                      colleges | 🥇 58.5% (0.4s) |  🥈 42.2% (6.6s) |    40.2% (15.1s) |
|                      diamonds | 🥇 98.2% (0.3s) | 🥈 95.2% (13.7s) |                / |
|                     elevators | 🥇 87.7% (0.5s) |  🥈 82.6% (6.5s) |                / |
|                     house_16h | 🥇 67.7% (0.4s) |  🥈 52.8% (6.0s) |                / |
|          house_prices_nominal | 🥇 89.0% (0.3s) |  🥈 78.3% (2.1s) |     -2.9% (1.2s) |
|                   house_sales | 🥇 89.2% (0.4s) |  🥈 77.8% (5.9s) |                / |
|           mip-2016-regression | 🥇 59.2% (0.4s) |  🥈 34.9% (5.8s) |    -27.3% (0.4s) |
|                     moneyball | 🥇 93.2% (0.3s) |  🥈 91.3% (1.1s) |      0.8% (0.2s) |
|                           pol | 🥇 98.7% (0.3s) |  🥈 74.9% (4.6s) |                / |
|                         quake |   -10.7% (0.2s) |  🥇 -1.0% (1.6s) | 🥈 -10.7% (0.1s) |
| sat11-hand-runtime-regression | 🥇 78.3% (0.4s) |  🥈 61.7% (2.1s) |    -56.3% (5.1s) |
|                       sensory | 🥇 29.2% (0.1s) |      3.0% (1.6s) |  🥈 16.4% (0.0s) |
|                        socmob | 🥇 79.6% (0.2s) |  🥈 72.5% (6.6s) |     30.8% (0.1s) |
|                      space_ga | 🥇 70.3% (0.3s) |  🥈 43.6% (1.5s) |     35.9% (0.2s) |
|                       tecator | 🥈 98.3% (0.1s) |  🥇 99.4% (0.9s) |     78.5% (0.0s) |
|                      us_crime | 🥈 62.8% (0.6s) |  🥇 63.0% (2.3s) |      6.7% (0.8s) |
|                  wine_quality | 🥇 45.6% (0.2s) |  🥈 36.5% (2.8s) |     16.4% (1.6s) |

</details>

## Contributing

<details>
<summary>Prerequisites</summary>

<details>
<summary>1. Set up Git to use SSH</summary>

1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
1. Configure SSH to automatically load your SSH keys:
    ```sh
    cat << EOF >> ~/.ssh/config
    Host *
      AddKeysToAgent yes
      IgnoreUnknown UseKeychain
      UseKeychain yes
    EOF
    ```

</details>

<details>
<summary>2. Install Docker</summary>

1. [Install Docker Desktop](https://www.docker.com/get-started).
    - Enable _Use Docker Compose V2_ in Docker Desktop's preferences window.
    - _Linux only_:
        - Export your user's user id and group id so that [files created in the Dev Container are owned by your user](https://github.com/moby/moby/issues/3206):
            ```sh
            cat << EOF >> ~/.bashrc
            export UID=$(id --user)
            export GID=$(id --group)
            EOF
            ```

</details>

<details>
<summary>3. Install VS Code or PyCharm</summary>

1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
2. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [configure PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

</details>

</details>

<details open>
<summary>Development environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).
1. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsorber/neo-ls-svm) to clone this repository in a container volume and create a Dev Container with VS Code.
1. _Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm_: clone this repository, open it with PyCharm, and [configure Docker Compose as a remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote) with the `dev` service.
1. _Terminal_: clone this repository, open it with your terminal, and run `docker compose up --detach dev` to start a Dev Container in the background, and then run `docker compose exec dev zsh` to open a shell prompt in the Dev Container.

</details>

<details>
<summary>Developing</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `poetry add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `poetry.lock`. Add `--group test` or `--group dev` to install a CI or development dependency, respectively.
- Run `poetry update` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag.

</details>
