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
X, y = fetch_openml("credit-g", version=1, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = make_pipeline(TableVectorizer(), NeoLSSVM())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 76.7% (compared to sklearn.svm.SVC's 70.7%)

# Regression example:
X, y = fetch_openml("ames_housing", version=1, return_X_y=True, as_frame=True, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = make_pipeline(TableVectorizer(), NeoLSSVM())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 81.8% (compared to sklearn.svm.SVR's -11.8%)
```

## Comparison of kernel method implementations

| Kernel method           | Unconstrained optimization | Classification / Regression | Large-scale  | Probabilistic | Hyperparameter optimization |
|-------------------------|----------------------------|-----------------------------|--------------|---------------|-----------------------------|
| [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) / [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) | ❌ | ✅/✅ | ❌ | ❌ | ❌ |
| [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) / [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html) + [Feature map](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation) | ❌ | ✅/✅ | ✅ | ❌ | ❌ |
| [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html) | ✅ | ❌/✅ | ❌ | ❌ | ❌ |
| [GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html) / [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) | ✅ | ✅/✅ | ❌ | Bayesian | Multi-step |
| [LS-SVMlab](https://www.esat.kuleuven.be/sista/lssvmlab/) | ✅ | ✅/✅ | ✅ | Bayesian | Multi-step |
| [NeoLSSVM](https://github.com/lsorber/neo-ls-svm) | ✅ | ✅/✅ | ✅ | Conformal | Single-step |

Other kernel methods not included in the comparison above for lack of a readily available implementation include [RBF Networks](https://en.wikipedia.org/wiki/Radial_basis_function_network) ("Kernel Ridge Regression without a bias term") and [Relevance Vector Machines](https://en.wikipedia.org/wiki/Relevance_vector_machines) ("a Bayesian Support Vector Machine").

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

|                       dataset |   GaussianProcessRegressor |      KernelRidge |   LGBMRegressor |        NeoLSSVM |              SVR |
|------------------------------:|---------------------------:|-----------------:|----------------:|----------------:|-----------------:|
|                       abalone |            -2059.2% (1.0s) |     54.1% (0.8s) | 🥈 56.2% (0.1s) | 🥇 59.5% (1.1s) |     51.3% (0.2s) |
|                        boston |             -679.3% (0.0s) |   -589.6% (0.1s) | 🥇 91.7% (0.2s) | 🥈 89.3% (0.3s) |     35.1% (0.0s) |
|              brazilian_houses |            -128.8% (10.3s) |   -128.8% (6.5s) | 🥈 55.9% (0.2s) | 🥇 88.3% (1.2s) |      5.4% (1.9s) |
|                      colleges |             -592.2% (9.3s) |   -592.2% (4.9s) | 🥇 58.5% (0.3s) | 🥈 42.4% (3.8s) |     40.2% (5.0s) |
|                      diamonds |                          / |                / | 🥇 98.2% (0.3s) | 🥈 95.2% (4.1s) |                / |
|                     elevators |                          / |                / | 🥇 87.7% (0.3s) | 🥈 82.6% (2.2s) |                / |
|                     house_16h |                          / |                / | 🥇 67.7% (0.3s) | 🥈 52.8% (2.0s) |                / |
|          house_prices_nominal |             -400.7% (0.3s) |   -399.1% (0.2s) | 🥇 89.0% (0.3s) | 🥈 78.2% (0.9s) |     -2.9% (0.3s) |
|                   house_sales |                          / |                / | 🥇 89.2% (0.3s) | 🥈 77.8% (1.9s) |                / |
|           mip-2016-regression |               12.2% (0.1s) |     12.3% (0.2s) | 🥇 59.2% (0.4s) | 🥈 34.9% (1.4s) |    -27.3% (0.1s) |
|                     moneyball |            -5735.6% (0.1s) |  -1036.9% (0.2s) | 🥇 93.2% (0.2s) | 🥈 91.2% (0.6s) |      0.8% (0.1s) |
|                           pol |                          / |                / | 🥇 98.7% (0.2s) | 🥈 75.2% (1.5s) |                / |
|                         quake |           -77137.1% (0.1s) | -79656.3% (0.2s) |   -10.7% (0.2s) | 🥇 -0.1% (0.5s) | 🥈 -10.7% (0.0s) |
| sat11-hand-runtime-regression |               57.7% (1.8s) |     60.2% (0.9s) | 🥇 78.3% (0.3s) | 🥈 61.7% (1.1s) |    -56.3% (1.0s) |
|                       sensory |            -3400.0% (0.0s) |   -143.8% (0.2s) | 🥇 29.2% (0.1s) |     3.8% (0.4s) |  🥈 16.4% (0.0s) |
|                        socmob |               11.7% (0.1s) |     26.7% (0.1s) | 🥇 79.6% (0.2s) | 🥈 72.5% (1.5s) |     30.8% (0.1s) |
|                      space_ga |             -833.2% (0.3s) |   -833.2% (0.2s) | 🥇 70.3% (0.3s) | 🥈 43.7% (0.6s) |     35.9% (0.1s) |
|                       tecator |             -109.0% (0.0s) |     94.1% (0.2s) | 🥈 98.3% (0.1s) | 🥇 99.4% (0.3s) |     78.5% (0.0s) |
|                      us_crime |              -12.5% (0.4s) |     37.3% (0.3s) | 🥈 62.8% (0.5s) | 🥇 63.0% (0.9s) |      6.7% (0.2s) |
|                  wine_quality |            -1774.5% (3.1s) |   -474.9% (2.0s) | 🥇 45.6% (0.2s) |    -8.0% (0.9s) |  🥈 16.4% (0.5s) |

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
