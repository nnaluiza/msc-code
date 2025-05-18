# Master's Project - Autoparametrization

The project aims to create an effective and scalable solution for the automatic parametrization of the Growing Neural Gas (GNG) network.

## Specifications

This project was created and developed using the following hardware and software configuration:

| Category           | Information                                 |
|--------------------|---------------------------------------------|
| Operating System   | Ubuntu 24.10.0 LTS                          |
| Processor          | Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz    |
| RAM                | 16 GB                                       |
| Python             | 3.12.7                                      |

These configurations provide the foundation for the development and execution of this project. Be sure to check the versions of the tools and operating system to ensure compatibility with the project.

## Managing Virtual Environments and Pre-Commit Hooks

### Virtual Environment Setup

| Task                      | Windows (PowerShell)                     | Ubuntu                            |
|---------------------------|------------------------------------------|-----------------------------------|
| Create virtual environment | `python -m venv env-windows`            | `python3.13 -m venv env-ubuntu`   |
| Activate environment      | `.\env-windows\Scripts\Activate.ps1`     | `source env-ubuntu/bin/activate`  |
| Install dependencies      | `pip install -r requirements.txt`        | `pip install -r requirements.txt` |
| Deactivate environment    | `deactivate`                             | `deactivate`                      |
| Run the code              | `python src/script.py`                   | `python src/script.py`            |

---