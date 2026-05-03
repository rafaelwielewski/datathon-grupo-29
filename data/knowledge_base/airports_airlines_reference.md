# Airports and Airlines Reference

## Major US Airports (by traffic)

| Code | City | State | Typical Delay Rate |
|------|------|-------|--------------------|
| ATL  | Atlanta | GA | ~20% |
| LAX  | Los Angeles | CA | ~19% |
| ORD  | Chicago O'Hare | IL | ~25% |
| DFW  | Dallas/Fort Worth | TX | ~18% |
| DEN  | Denver | CO | ~17% |
| JFK  | New York | NY | ~22% |
| SFO  | San Francisco | CA | ~24% |
| LAS  | Las Vegas | NV | ~14% |
| SEA  | Seattle | WA | ~16% |
| MCO  | Orlando | FL | ~15% |

O'Hare (ORD) and San Francisco (SFO) have the highest delay rates due to weather and congestion.

## Major US Airlines

| Code | Name | Typical Delay Rate |
|------|------|--------------------|
| AA   | American Airlines | ~20% |
| DL   | Delta Air Lines | ~17% |
| UA   | United Airlines | ~21% |
| WN   | Southwest Airlines | ~18% |
| AS   | Alaska Airlines | ~15% |
| B6   | JetBlue | ~23% |
| NK   | Spirit Airlines | ~26% |
| F9   | Frontier Airlines | ~24% |

Alaska Airlines consistently ranks among the most on-time carriers.
Spirit and Frontier have the highest delay rates, often linked to tight turnaround schedules.

## Interpreting Model Output
- `delayed: true, probability: 0.75, confidence: high` → strong signal, plan for delays
- `delayed: false, probability: 0.30, confidence: medium` → moderate confidence in on-time arrival
- `confidence: low` → probability close to threshold; outcome is uncertain
