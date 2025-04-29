# Constants
GRAVITY = 9.80665  # standard gravitational acceleration (m/s)
r_earth = 6370000.0  # Radius of Earth (m)
Omega = 7.2921159e-5  # Angular velocity of Earth (Rad/s)
Rd = 287.04  # R for dry air (J/kg/K)
Rv = 461.50  # R for water vapor
Cpd = 1005.0  # Specific heat of dry air at constant pressure (J/kg/K)
Cl = 4186.0  # Specific heat of liquid water (J/kg/K)
Lv0 = 2.501e6  # Latent heat of vaporization for water at 0 Celsius (J/kg)
VON_KARMAN = 0.41  # von Karman constant
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
SOLAR_CONSTANT = 1361  # W/m² (Extraterrestrial solar radiation)

# Reasonable net radiation ranges
DAYTIME_RANGE = (50, 1000)  # Expected range in W/m²
NIGHTTIME_RANGE = (-200, 100)  # Expected range in W/m²
