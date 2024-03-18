
# IHSetYates09
Python package to run and calibrate Yates et al. (2009) equilibrium-based shoreline evolution model.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetYates09.git

```

---
## :zap: Main methods

* [yates09](./IHSetYates09/yates09.py):
```python
# model's it self
yates09(E, dt, a, b, cacr, cero, Yini, vlt)
```
* [cal_Yates09](./IHSetYates09/calibration.py):
```python
# class that prepare the simulation framework
cal_Yates09(path)
```



## :package: Package structures
````

IHSetYates09
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetYates09
│   ├── calibration.py
│   └── yates09.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).
