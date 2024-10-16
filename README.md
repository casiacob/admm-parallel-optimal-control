## Jax implementation of ADMM constrained parallel optimal control

## First clone and install the unconstrained parallel optimal control solver
Clone the repository:

```
$ git clone https://github.com/casiacob/parallel-optimal-control.git
```

Create conda environment:
```
$ conda create --name paroc python=3.10
$ conda activate paroc
$ cd parallel-optimal-control
$ pip install .
```
## ADMM
Clone the repository
```
$ cd ..
$ git clone https://github.com/casiacob/admm-parallel-optimal-control.git
$ cd admm-parallel-optimal-control
$ pip install .
```
Constrained pendulum runtime example (requires GPU)
```
$ cd examples
$ python pendulum_runtime.py
```
Constrained cartpole runtime example (requires GPU)
```
$ cd examples
$ python cartpole_runtime.py
```

