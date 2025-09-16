# Installation and setup guide 

This exercise contains information on how to install and use conda and VS code through the course. 

## Install Conda and VS Code

We strongly recommend to use Conda (miniforge) for your Python installation and other tools. If you do not have it installed already follow the guides at [Python Support](https://pythonsupport.dtu.dk/install/python.html) for your operating system. 

To edit the code given in this course we use VS Code, but if you are more familiar with other editors then feel free to use them. However, the TAs might not be able to help with editor(IDE) specific questions. If you do not have a working installation of VS Code, we recommend downloading it from [VS Code](https://code.visualstudio.com/download) and chose your operating system.  


## Virtual environments

When you have installed Conda and VS Code, you can start by creating a **virtual environment**. A virtual environment is a practical way of keeping a specific Python interpreter, installed libraries and scrips isolated. So if you have several projects that each requires different libraries, they can be kept separate.

### Creating and activating a conda environment
Start an **VS Code terminal** and do:

```Shell
conda create --name course02503 python=3.10
conda activate course02503
```

here a new virtual environment called `course02503` is created and activated. If you want more details on how setup environments there is a guide from [Python support](https://pythonsupport.dtu.dk/learn-more/packages-and-environments/environments.html). 



## Python scripts or Notebooks

In this course, there are basically two ways that you can do the exercises:

1. **Jupyter notebooks** is a notebook file with a mix of text(Markdown text) snippets and code blocks, where you can run Python code. We recommend using [Visual Studio Code](https://code.visualstudio.com/) to open and edit the notebooks. For more information and tutorial we recommed looking at [Python Support](https://pythonsupport.dtu.dk/learn-more/vscode/jupyter-notebooks.html). 
2. **Python scripts** are text files with the `.py` ending (for example `myscript.py`). Python scripts are normally edited and executed using a dedicated **code editor (IDE)**. We recommend [Visual Studio Code](https://code.visualstudio.com/) or alternatively use [Spyder](https://www.spyder-ide.org/) or [PyCharm](https://www.jetbrains.com/products/compare/?product=pycharm&product=pycharm-ce). 


We provide the exercise code as pieces of Python code and it is your decision if you want to use them in a Notebook or in a script.

Please see [Figures in notebooks](FigsInNotebooks) for details on figures in notebooks.

If you need an introduction or a refresher to Jupyter notebooks, Python, NumPY and Matplotlib, you can find a good introduction at: [Introduction to Jupyter notebooks and Python](https://github.com/nabriis/Jupyter_Python_Introduction).

It is important that your Notebook, VS Code, Spyder, Pycharm or other development tool is set to use the virtual environment that you created before. To select the right enviroment in VS Code: 
1. Press crtl + shift + P
2. Select Python: Select Interpreter and press enter 
3. Choose the option with the ending (`course02503`) 



For checking this in Jupyter Notebooks run the following commands in code cell: 

```Shell
import sys
print(sys.exec_prefix)
```


## Installing Python packages

In this course, we will use several Python packages. We will install them as needed. To install the package [scikit-image (skimage)](https://scikit-image.org/), start an **VS Code terminal** and do:

```Shell
conda activate course02503
conda install scikit-image
```

Further info can be found here: [sci-kit image](https://scikit-image.org/docs/0.25.x/user_guide/install.html).

You should also install 
- [matplotlib](https://anaconda.org/conda-forge/matplotlib)
- [pydicom](https://anaconda.org/conda-forge/pydicom)


## Exercise data and material (Setting up)

The data and material needed for this exercise can be found here: [exercise data and material](data)

Start by creating and exercise folder where you keep your data, Python scripts or Notebooks. Download or clone the data and material (from GitHub instructions below) and place them in this folder.

### Using GitHub to get all the material and data at once

You can get all the exercise material including the data by (for example):
- Option 1: download the current version of the exercise and data as a .zip file. Do this by going to the [main page](https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main) and pressing the green `<>Code` button and select the **Download ZIP** option.
- Option 2: Clone the repository using a [Git](https://git-scm.com/) client.
  1. Go to terminal and type
  2. ```Shell
     cd /path/to/exercise/folder
     git clone https://github.com/RasmusRPaulsen/DTUImageAnalysis.git
     ```


