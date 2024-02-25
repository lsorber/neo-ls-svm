[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsorber/neo-ls-svm) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=739301655)

# Neo LS-SVM

Neo LS-SVM is a modern [Least-Squares Support Vector Machine](https://en.wikipedia.org/wiki/Least-squares_support_vector_machine) implementation in Python that offers several benefits over sklearn's classic `sklearn.svm.SVC` classifier and `sklearn.svm.SVR` regressor:

1. ⚡ Linear complexity in the number of training examples with [Orthogonal Random Features](https://arxiv.org/abs/1610.09072).
2. 🚀 Hyperparameter free: zero-cost optimization of the regularisation parameter γ and kernel parameter σ.
3. 🏔️ Adds a new tertiary objective that minimizes the complexity of the prediction surface.
4. 🎁 Returns the leave-one-out residuals and error for free after fitting.
5. 🌀 Learns an affine transformation of the feature matrix to optimally separate the target's bins.
6. 🪞 Can solve the LS-SVM both in the primal and dual space.
7. 🌡️ Isotonically calibrated `predict_proba` based on the leave-one-out predictions.
8. 🎲 Asymmetric conformal Bayesian confidence intervals for classification and regression.

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

### Confidence intervals

Neo LS-SVM implements conformal prediction with a Bayesian nonconformity estimate to compute confidence intervals for both classification and regression. Example usage:

```python
from neo_ls_svm import NeoLSSVM
from pandas import get_dummies
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load a regression problem and split in train and test.
X, y = fetch_openml("ames_housing", version=1, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(get_dummies(X), y, test_size=50, random_state=42)

# Fit a Neo LS-SVM model.
model = NeoLSSVM().fit(X_train, y_train)

# Predict the house prices and confidence intervals on the test set.
ŷ = model.predict(X_test)
ŷ_conf = model.predict_proba(X_test, confidence_interval=True, confidence_level=0.95)
# ŷ_conf[:, 0] and ŷ_conf[:, 1] are the lower and upper bound of the confidence interval for the predictions ŷ, respectively
```

Let's visualize the confidence intervals on the test set:

<img src="https://github.com/lsorber/neo-ls-svm/assets/4543654/472bf358-34d7-4a1a-8b5c-595fe65dbf77" width="512">

<details>
<summary>Expand to see the code that generated the above graph.</summary>

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

idx = np.argsort(-ŷ)
y_ticks = np.arange(1, len(X_test) + 1)
plt.figure(figsize=(4, 5))
plt.barh(y_ticks, ŷ_conf[idx, 1] - ŷ_conf[idx, 0], left=ŷ_conf[idx, 0], label="95% Confidence interval", color="lightblue")
plt.plot(y_test.iloc[idx], y_ticks, "s", markersize=3, markerfacecolor="none", markeredgecolor="cornflowerblue", label="Actual value")
plt.plot(ŷ[idx], y_ticks, "s", color="mediumblue", markersize=0.6, label="Predicted value")
plt.xlabel("House price")
plt.ylabel("Test house index")
plt.yticks(y_ticks, y_ticks)
plt.tick_params(axis="y", labelsize=6)
plt.grid(axis="x", color="lightsteelblue", linestyle=":", linewidth=0.5)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend()
plt.tight_layout()
plt.show()
```
</details>

## Benchmarks

We select all binary classification and regression datasets below 1M entries from the [AutoML Benchmark](https://arxiv.org/abs/2207.12560). Each dataset is split into 85% for training and 15% for testing. We apply `skrub.TableVectorizer` as a preprocessing step for `neo_ls_svm.NeoLSSVM` and `sklearn.svm.SVC,SVR` to vectorize the pandas DataFrame training data into a NumPy array. Models are fitted only once on each dataset, with their default settings and no hyperparameter tuning.

<details open>
<summary>Binary classification</summary>

ROC-AUC on 15% test set:

|                          dataset |   LGBMClassifier |        NeoLSSVM |             SVC |
|---------------------------------:|-----------------:|----------------:|----------------:|
|                              ada |  🥈 90.9% (0.1s) | 🥇 90.9% (0.8s) |    83.1% (1.0s) |
|                            adult |  🥇 93.0% (0.5s) | 🥈 89.1% (6.0s) |               / |
|           amazon_employee_access |  🥇 85.6% (0.5s) | 🥈 64.5% (2.8s) |               / |
|                           arcene |  🥈 78.0% (0.6s) |    70.0% (4.4s) | 🥇 82.0% (3.4s) |
|                       australian |  🥇 88.3% (0.2s) |    79.9% (0.4s) | 🥈 81.9% (0.0s) |
|                   bank-marketing |  🥇 93.5% (0.3s) | 🥈 91.0% (4.1s) |               / |
| blood-transfusion-service-center |     62.0% (0.1s) | 🥇 71.0% (0.5s) | 🥈 69.7% (0.0s) |
|                            churn |  🥇 91.7% (0.4s) | 🥈 81.0% (0.8s) |    70.6% (0.8s) |
|           click_prediction_small |  🥇 67.7% (0.4s) | 🥈 66.6% (3.3s) |               / |
|                          jasmine |  🥇 86.1% (0.3s) |    79.5% (1.2s) | 🥈 85.3% (1.8s) |
|                              kc1 |  🥇 78.9% (0.2s) | 🥈 76.6% (0.5s) |    45.7% (0.2s) |
|                         kr-vs-kp | 🥇 100.0% (0.2s) |    99.2% (0.8s) | 🥈 99.4% (0.6s) |
|                         madeline |  🥇 93.1% (0.4s) |    65.6% (0.8s) | 🥈 82.5% (4.5s) |
|                  ozone-level-8hr |  🥈 91.2% (0.3s) | 🥇 91.6% (0.7s) |    72.8% (0.2s) |
|                              pc4 |  🥇 95.3% (0.3s) | 🥈 90.9% (0.5s) |    25.7% (0.1s) |
|                 phishingwebsites |  🥇 99.5% (0.3s) | 🥈 98.9% (1.3s) |    98.7% (2.6s) |
|                          phoneme |  🥇 95.6% (0.2s) | 🥈 93.5% (0.8s) |    91.2% (0.7s) |
|                      qsar-biodeg |  🥇 92.7% (0.2s) | 🥈 91.1% (1.2s) |    86.8% (0.1s) |
|                        satellite |  🥈 98.7% (0.2s) | 🥇 99.5% (0.8s) |    98.5% (0.1s) |
|                          sylvine |  🥇 98.5% (0.2s) | 🥈 97.1% (0.8s) |    96.5% (1.0s) |
|                             wilt |  🥈 99.5% (0.2s) | 🥇 99.8% (0.9s) |    98.9% (0.2s) |

</details>

<details open>
<summary>Regression</summary>

R² on 15% test set:

|                       dataset |   LGBMRegressor |        NeoLSSVM |              SVR |
|------------------------------:|----------------:|----------------:|-----------------:|
|                       abalone | 🥈 56.2% (0.1s) | 🥇 59.5% (1.1s) |     51.3% (0.2s) |
|                        boston | 🥇 91.7% (0.2s) | 🥈 89.3% (0.4s) |     35.1% (0.0s) |
|              brazilian_houses | 🥈 55.9% (0.4s) | 🥇 88.3% (1.5s) |      5.4% (2.0s) |
|                      colleges | 🥇 58.5% (0.4s) | 🥈 43.7% (4.1s) |     40.2% (5.1s) |
|                      diamonds | 🥇 98.2% (0.7s) | 🥈 95.2% (4.5s) |                / |
|                     elevators | 🥇 87.7% (0.4s) | 🥈 82.6% (2.6s) |                / |
|                     house_16h | 🥇 67.7% (0.3s) | 🥈 52.8% (2.4s) |                / |
|          house_prices_nominal | 🥇 89.0% (0.6s) | 🥈 78.2% (1.3s) |     -2.9% (0.3s) |
|                   house_sales | 🥇 89.2% (1.3s) | 🥈 77.8% (2.2s) |                / |
|           mip-2016-regression | 🥇 59.2% (0.4s) | 🥈 34.9% (2.6s) |    -27.3% (0.1s) |
|                     moneyball | 🥇 93.2% (0.2s) | 🥈 91.2% (0.6s) |      0.8% (0.1s) |
|                           pol | 🥇 98.7% (0.3s) | 🥈 75.2% (1.7s) |                / |
|                         quake |   -10.7% (0.2s) | 🥇 -0.1% (0.5s) | 🥈 -10.7% (0.0s) |
| sat11-hand-runtime-regression | 🥇 78.3% (0.5s) | 🥈 61.7% (1.0s) |    -56.3% (1.0s) |
|                       sensory | 🥇 29.2% (0.2s) |     3.8% (0.4s) |  🥈 16.4% (0.0s) |
|                        socmob | 🥇 79.6% (0.2s) | 🥈 72.5% (1.5s) |     30.8% (0.0s) |
|                      space_ga | 🥇 70.3% (0.2s) | 🥈 43.7% (0.6s) |     35.9% (0.1s) |
|                       tecator | 🥈 98.3% (0.1s) | 🥇 99.4% (0.2s) |     78.5% (0.0s) |
|                      us_crime | 🥈 62.8% (0.4s) | 🥇 63.0% (0.8s) |      6.7% (0.2s) |
|                  wine_quality | 🥇 45.6% (0.6s) |    -8.0% (0.9s) |  🥈 16.4% (0.5s) |

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
