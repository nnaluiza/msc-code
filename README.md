# Master's Project - Autoparametrization

## Specifications

This project was created and developed using the following hardware and software configuration:

| Category           | Information                                 |
|--------------------|---------------------------------------------|
| Operating System   | Ubuntu 24.10.0 LTS                          |
| Processor          | Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz    |
| RAM                | 16 GB                                       |
| Python             | 3.12.7                                      |

These configurations provide the foundation for the development and execution of this project. Be sure to check the versions of the tools and operating system to ensure compatibility with the project.

## Experiment Script Parameters
The experiment script requires specific parameters to be passed in the following order:

- seed: A number to initialize random processes for reproducibility;
- size: A number specifying the Working Memory size;
- reps: A number indicating the number of repetitions;
- distance metric: A string specifying the distance metric;
- dataset_name: A string specifying the dataset name;

The parameters `seed`, `size`, and `reps` must be numbers greater than 0 to function correctly in the experiment script.

The distance metric parameter, however, is restricted to three specific options: "euclidean", "cosine", or "cityblock".

The allowed datasets that can be loaded are:
- Python datasets: iris, moons, blobs, and circles;
- Toy datasets: 2sp2glob.arff, 3-spiral.arff, chainlink.arff, complex8.arff, complex9.arff, diamond9.arff, ds2c2sc13.arff, hypercube.arff;
- Real datasets:

> [!IMPORTANT]
> On Windows, the command to run the experiment script is `python script.py`, while on Ubuntu, it is `python3 script.py`.
> This difference arises because Windows typically associates the python command with the Python executable in the system's PATH or the active virtual environment, whereas Ubuntu often has both Python 2 and Python 3 installed, requiring python3 to explicitly invoke Python 3 to ensure compatibility.

Example command:
```bash
Windows: python script.py 1 1000 30 euclidean iris
Ubuntu: python3 script.py 1 1000 30 euclidean iris
```

Any modifications to the experiment should be made exclusively in the experiment.sh file, as other files must remain unchanged to maintain the integrity of the experiment setup.

> [!CAUTION]
> The script name script.py must not be changed, as it is explicitly referenced in the experiment setup and configuration.

## Running Experiments

After setting up the environment, install the required dependencies from the `requirements.txt` file. Use the command:

```bash
pip install -r requirements.txt
```

> [!WARNING]
> For Ubuntu, you may need to override permissions due to system package restrictions. In this case, add the flag `--break-system-packages` at the end of the command.
> Additionally, ensure the script has executable permissions (e.g., `chmod +x experiment.sh` on Ubuntu) before running it.

After the setup is complete, run the experiment script to execute your experiment or automate additional tasks. Run the command:

```bash
./experiments.sh
```

Once executed, the code will run automatically without the need for further intervention.

## Managing Virtual Environments

Before using the commands in the table below, ensure that the `venv` library is installed, as it may not be included in some Python installations. To install it, run the following command:

```bash
pip install virtualenv
```

The table below provides the specific commands for each step on Windows and Ubuntu. Follow the commands in the order listed to set up and manage your virtual environment effectively.

### Virtual Environment Setup

| Task                      | Windows (PowerShell)                     | Ubuntu                            |
|---------------------------|------------------------------------------|-----------------------------------|
| Create virtual environment | `python -m venv env-windows`            | `python3.13 -m venv env-ubuntu`   |
| Activate environment      | `.\env-windows\Scripts\Activate.ps1`     | `source env-ubuntu/bin/activate`  |
| Install dependencies      | `pip install -r requirements.txt`        | `pip install -r requirements.txt` |
| Deactivate environment    | `deactivate`                             | `deactivate`                      |
| Run the code              | `python src/script.py`                   | `python src/script.py`            |



To run the code, just replicate the commands inside the virtual environment.

---