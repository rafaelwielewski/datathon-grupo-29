# Flight Delay Overview

## What is a flight delay?
A flight is considered delayed when it arrives at the gate 15 or more minutes after its scheduled arrival time.
The US DOT uses this 15-minute threshold for official statistics.

## Main causes of delays
- **Carrier delays**: mechanical problems, crew availability, aircraft cleaning.
- **Late aircraft**: the incoming aircraft arrived late from a previous flight.
- **Weather**: thunderstorms, fog, ice, snow affecting departures or arrivals.
- **National Air System (NAS)**: heavy traffic volume, air traffic control.
- **Security**: terminal evacuations or screening delays.

## Delay rates by time of day
Early morning flights (5-8 AM) have the lowest delay rates (~12%) because aircraft start fresh.
Evening flights (after 6 PM) accumulate cascade delays throughout the day, reaching 28-35%.

## Seasonal patterns
Delays peak in June-July (summer thunderstorm season) and December-January (winter storms).
The lowest delay months are September-October.

## Route characteristics
Short-haul routes (< 500 mi) show higher delay variability due to less buffer time in schedules.
Long-haul routes (> 1500 mi) have more schedule padding and generally lower delay rates.

## Model prediction
The CatBoost model predicts binary delay (arrival >= 15 minutes late) using pre-flight information:
flight schedule, route, airline, origin/destination airports, day of week, month, and historical delay rates.
The model outputs a probability and compares it to a precision-constrained threshold (~0.607).
