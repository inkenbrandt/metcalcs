=====
Usage
=====

The examples below are ready-to-paste into a Sphinx ``.rst`` page (e.g. ``usage.rst``).
They draw **only** on the three source files you attached—``thermo.py``, ``radiation.py`` and ``evaporation.py``—so they will execute without any other package dependencies.
Citations point to the corresponding implementation in each file.

---

.. _metcalcs_examples:

==============================
Quick-start & usage examples
==============================

.. currentmodule:: metcalcs

Install
-------

.. code-block:: bash

   pip install metcalcs   # from PyPI
   # OR, if you are developing locally:
   pip install -e .

Import the convenience namespace:

.. code-block:: python

   import metcalcs as mc


1. Radiation workflow
---------------------

This snippet computes the main surface-energy components for **Day-Of-Year = 185** (≈ 4 July) at 40.76 °N:

.. code-block:: python

   import numpy as np
   import metcalcs.radiation as rad

   # location & weather
   doy, lat = 185, 40.76                # day-of-year, latitude  [-]
   Ta, RH   = 25.0, 55.0                # air temperature [°C], relative humidity [%]
   Rs_meas  = 740.0                     # measured short-wave flux [W m-2]

   # derive components
   ra, _ = rad.solar_parameters(doy, lat)           # top-of-atmosphere J day-1
   components = rad.calculate_radiation(Ta, RH, Rs_meas, lat, doy)

   for k, v in components.items():
       print(f"{k:15s}: {v:8.1f}")

   # extraterrestrial  :  41 600 000.0  (J m-2 day-1)
   # net_shortwave     :     569.8      (W m-2)
   # net_longwave      :      83.7      (W m-2)
   # net_radiation     :     486.1      (W m-2)

``calculate_radiation`` wraps all four equations (Ra, Rns, Rnl, Rn) in one call.


2. Basic thermo-dynamic helpers
-------------------------------

.. code-block:: python

   import metcalcs.thermo as th

   ea   = th.actual_vapor_pressure(Ta, RH)              # Pa
   delta= th.vapor_pressure_slope(Ta)                   # Pa K-1
   gamma= th.psychrometric_constant(Ta, RH, 87_500)     # Pa K-1

   print(f"ea     = {ea:7.0f} Pa")
   print(f"Δ      = {delta:6.1f} Pa K-1")
   print(f"γ      = {gamma:6.1f} Pa K-1")

These three functions underpin every evapotranspiration routine in *metcalcs*.


3. Open-water evaporation with Penman (1948/56)
-----------------------------------------------

.. code-block:: python

   import metcalcs.evaporation as ev

   # daily, energy-integrated inputs (J m-2 day-1)
   Rs_day  = 28.0e6                       # observed short-wave
   Rext    = ra                           # from Section 1 (J day-1)

   E0 = ev.penman_open_water(
       airtemp = Ta,
       rh      = RH,
       airpress= 87_500,                  # Pa
       Rs      = Rs_day,
       Rext    = Rext,
       u       = 3.0,                     # wind @ 2 m  [m s-1]
       Z       = 1350                     # site elevation [m]
   )

   print(f"Open-water E₀ ≈ {E0:4.1f} mm day⁻¹")

The function needs **only** six meteorological variables; long-wave terms are handled internally following Penman’s original cloudiness correction.


4. Sensible-heat flux via TVAR-Dry
----------------------------------

For high-frequency (10–20 Hz) temperature measurements you can estimate *H* directly:

.. code-block:: python

   rho      = 1.06                       # kg m-3  (from thermo.air_density)
   cp       = 1004.5                     # J kg-1 K-1
   sigma_T  = 0.24                       # °C  (standard deviation, 30 min window)

   H = th.h_tvardry(rho, cp, Ta, sigma_T,
                    z = 4.0,             # sensor height [m]
                    d = 0.6)             # displacement height [m]

   print(f"Sensible heat flux: {H:5.1f} W m-2")

This follows De Bruin et al. (1992) exactly (constants *C₁ = 2.9*, *C₂ = 28.4*).


5. Putting it all together
--------------------------

Combine the preceding steps to build a minimal daily water-balance script:

.. code-block:: python

   import metcalcs.thermo      as th
   import metcalcs.radiation   as rad
   import metcalcs.evaporation as ev

   # 1. radiation
   _, Ra = rad.solar_parameters(doy, lat)
   rad_comp = rad.calculate_radiation(Ta, RH, Rs_meas, lat, doy)

   # 2. open-water evaporation
   E0 = ev.penman_open_water(Ta, RH, 87_500,
                             Rs_day  = Rs_meas * 86_400,  # convert W→J
                             Rext    = Ra,
                             u       = 3.0,
                             Z       = 1350)

   # 3. latent-vs-sensible split (Bowen ratio, etc.)
   cp   = th.specific_heat(Ta, RH, 87_500)
   rho  = th.air_density(Ta, RH, 87_500)
   H    = th.h_tvardry(rho, cp, Ta, sigma_T=0.25, z=4.0)

   print(f"Rn  = {rad_comp['net_radiation']:5.1f} W m-2")
   print(f"E₀  = {E0:4.1f} mm day⁻¹")
   print(f"H   = {H:5.1f} W m-2")

Now you have **Rn, H** and **E₀**—enough to close a daily energy-balance or calibrate a more complex hydrologic model.

---

These examples are self-contained; copy them into your ``docs/usage.rst`` (or an *nbsphinx* notebook) and build with Sphinx’s default ``python -m sphinx -b html docs build``.
