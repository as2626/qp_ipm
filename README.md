```
git clone --recurse-submodules git@github.com:as2626/qp_ipm.git \
&& mkdir build
```

```
cd eigen \
&& mkdir build \
&& cd build \
&& cmake .. \
&& make install
```

N.B., much credit goes to [Govind Chari](https://github.com/govindchari/QPSolver) for a reference implementation.
