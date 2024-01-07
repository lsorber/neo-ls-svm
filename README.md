[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsorber/neo-ls-svm)

# Neo LS-SVM

Neo LS-SVM is a modern [least-squares support vector machine](https://en.wikipedia.org/wiki/Least-squares_support_vector_machine) implementation in Python that offers several benefits over sklearn's classic `sklearn.svm.SVC` classifier and `sklearn.svm.SVR` regressor:

1. ⚡ Linear complexity in the number of training examples with [Orthogonal Random Features](https://arxiv.org/abs/1610.09072).
2. 🚀 Hyperparameter free: zero-cost optimization of the regularisation parameter γ and kernel parameter σ.
3. 🏔️ Adds a new tertiary objective that minimizes the complexity of the prediction surface.
4. 🎁 Returns the leave-one-out residuals, leverage, and error for free after fitting.
5. 🌀 Learns an affine transformation of the feature matrix to optimally separate the target's bins.
6. 🪞 Can solve the LS-SVM both in the primal and dual space.
7. 🌡️ Isotonically calibrated `predict_proba` based on the leave-one-out predictions.

## Using

First, install this package with:
```bash
pip install neo-ls-svm
```

Then, you can import `neo_ls_svm.NeoLSSVM` as an sklearn-compatible binary classifier and regressor. Example usage:

```python
from neo_ls_svm import NeoLSSVM
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer  # Vectorizes a pandas DataFrame into a NumPy array.

# Binary classification example:
X, y = fetch_openml("credit-g", return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = make_pipeline(TableVectorizer(), NeoLSSVM())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 77.3% (compared to sklearn.svm.SVC's 70.7%)

# Regression example:
X, y = fetch_openml("ames_housing", return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = make_pipeline(TableVectorizer(), NeoLSSVM())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 82.0% (compared to sklearn.svm.SVR's -11.8%)
```

## Benchmarks

We select all binary classification and regression datasets below 1M entries from the [AutoML Benchmark](https://arxiv.org/abs/2207.12560). Each dataset is split into 85% for training and 15% for testing. We apply `skrub.TableVectorizer` as a preprocessing step for `neo_ls_svm.NeoLSSVM` and `sklearn.svm.SVC,SVR` to vectorize the pandas DataFrame training data into a NumPy array. Models are fitted only once on each dataset, with their default settings and no hyperparameter tuning.

<details open>
<summary>Binary classification</summary>

ROC-AUC on 15% test set:

|                          dataset |   LGBMClassifier |        NeoLSSVM |             SVC |
|---------------------------------:|-----------------:|----------------:|----------------:|
|                              ada |  🥈 90.9% (0.2s) | 🥇 90.9% (1.1s) |    83.1% (1.1s) |
|                            adult |  🥇 93.0% (1.9s) | 🥈 89.0% (6.5s) |               / |
|           amazon_employee_access |  🥇 85.6% (1.0s) | 🥈 64.5% (3.9s) |               / |
|                           arcene |  🥈 78.0% (0.7s) |    66.0% (6.4s) | 🥇 82.0% (3.4s) |
|                       australian |  🥇 88.3% (0.3s) |    80.2% (0.6s) | 🥈 81.9% (0.0s) |
|                   bank-marketing |  🥇 93.5% (0.8s) | 🥈 91.0% (5.5s) |               / |
| blood-transfusion-service-center |     62.0% (0.2s) | 🥇 69.9% (0.5s) | 🥈 69.7% (0.0s) |
|                            churn |  🥇 91.7% (0.9s) | 🥈 81.0% (1.4s) |    70.6% (0.8s) |
|           click_prediction_small |  🥇 67.7% (1.0s) | 🥈 66.6% (4.5s) |               / |
|                          jasmine |  🥇 86.1% (0.5s) |    79.7% (1.2s) | 🥈 85.3% (1.8s) |
|                              kc1 |  🥇 78.9% (0.4s) | 🥈 76.6% (0.7s) |    45.7% (0.2s) |
|                         kr-vs-kp | 🥇 100.0% (0.6s) |    99.2% (1.0s) | 🥈 99.4% (0.6s) |
|                         madeline |  🥇 93.1% (1.0s) |    64.9% (1.2s) | 🥈 82.5% (4.6s) |
|                  ozone-level-8hr |  🥈 91.2% (0.6s) | 🥇 91.6% (1.0s) |    72.8% (0.2s) |
|                              pc4 |  🥇 95.3% (0.5s) | 🥈 90.9% (0.6s) |    74.3% (0.1s) |
|                 phishingwebsites |  🥇 99.5% (0.5s) | 🥈 98.9% (1.9s) |    98.7% (2.7s) |
|                          phoneme |  🥇 95.6% (0.4s) | 🥈 93.5% (1.1s) |    91.2% (0.7s) |
|                      qsar-biodeg |  🥇 92.7% (0.4s) | 🥈 90.7% (0.7s) |    86.8% (0.1s) |
|                        satellite |  🥈 98.7% (0.4s) | 🥇 99.5% (1.1s) |    98.5% (0.1s) |
|                          sylvine |  🥇 98.5% (0.3s) | 🥈 97.1% (1.0s) |    96.5% (1.0s) |
|                             wilt |  🥈 99.5% (0.3s) | 🥇 99.8% (1.0s) |    98.9% (0.2s) |

</details>

<details open>
<summary>Regression</summary>

R² on 15% test set:

|                       dataset |   LGBMRegressor |        NeoLSSVM |              SVR |
|------------------------------:|----------------:|----------------:|-----------------:|
|                       abalone | 🥈 56.2% (0.2s) | 🥇 59.5% (1.4s) |     51.3% (0.2s) |
|                        boston | 🥇 91.7% (0.4s) | 🥈 87.9% (0.6s) |     35.1% (0.0s) |
|              brazilian_houses | 🥈 55.9% (0.6s) | 🥇 88.3% (1.8s) |      5.4% (2.0s) |
|                      colleges | 🥇 58.5% (0.5s) | 🥈 42.6% (4.3s) |     40.2% (5.3s) |
|                      diamonds | 🥇 98.2% (0.4s) | 🥈 95.2% (5.5s) |                / |
|                     elevators | 🥇 87.7% (0.4s) | 🥈 82.6% (3.0s) |                / |
|                     house_16h | 🥇 67.7% (0.4s) | 🥈 52.8% (2.7s) |                / |
|          house_prices_nominal | 🥇 89.0% (0.3s) | 🥈 78.2% (1.1s) |     -2.9% (0.3s) |
|                   house_sales | 🥇 89.2% (0.5s) | 🥈 77.8% (2.6s) |                / |
|           mip-2016-regression | 🥇 59.2% (0.5s) | 🥈 32.5% (0.8s) |    -27.3% (0.1s) |
|                     moneyball | 🥇 93.2% (0.2s) | 🥈 91.2% (0.7s) |      0.8% (0.1s) |
|                           pol | 🥇 98.7% (0.4s) | 🥈 75.2% (2.3s) |                / |
|                         quake |   -10.7% (0.3s) | 🥇 -0.1% (0.9s) | 🥈 -10.7% (0.0s) |
| sat11-hand-runtime-regression | 🥇 78.3% (0.5s) | 🥈 61.7% (1.3s) |    -56.3% (1.0s) |
|                       sensory | 🥇 29.2% (0.2s) |     3.7% (0.5s) |  🥈 16.4% (0.0s) |
|                        socmob | 🥇 79.6% (0.2s) | 🥈 70.7% (0.6s) |     30.8% (0.0s) |
|                      space_ga | 🥇 70.3% (0.4s) | 🥈 43.7% (0.8s) |     35.9% (0.1s) |
|                       tecator | 🥈 98.3% (0.2s) | 🥇 99.3% (0.6s) |     78.5% (0.0s) |
|                      us_crime | 🥈 62.8% (0.6s) | 🥇 63.0% (1.2s) |      6.7% (0.2s) |
|                  wine_quality | 🥇 45.6% (0.3s) |    -7.8% (1.3s) |  🥈 16.4% (0.5s) |

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
