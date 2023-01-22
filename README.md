# Over-and-under-estimation-

This work aims to model the spread of
the coronavirus in the state of Israel
at the beginning of the first vaccination campaign,
using 10 different scenarios of mixing between vaccinated
and non-vaccinated individuals. The code runs simultaneously
for one million particles and is divided into 1578 static regions.
The "statistical_area" variable divides the space into different statistical regions,
with each region's column 1 indicating the serial number of the first particle in that
region and column 2 indicating the last particle. The input for all scenarios is the
vaccination rate of the population at the resolution of a statistical area, with the
scenarios differing in the degree of mixing between vaccinated and non-vaccinated individuals.
The "t_rand" variable contains the day on which each particle receives their second vaccine
and the general vaccination rate of the area is determined by the true vaccination rate
(official information from the State of Israel). The simulation begins on November 22, 2020,
and the widespread vaccination of the population begins on December 24th.
For each scenario, the number of infected individuals who are vaccinated and unvaccinated
is analyzed on the same day and the vaccine efficacy for each mixing index I is calculated based on this information.
