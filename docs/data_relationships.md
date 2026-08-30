# Data Relationships

## Relationship Map

```
EMPLOYEE (employee_attrition_processed.csv)
   |
   +-- EmployeeID --- ENGAGEMENT (engagement_processed.csv)
   |                  [0% overlap - different ID spaces]
   |
   +-- JobRole ------ OCCUPATION (occupation_master.csv)
                      [0% overlap - different vocabularies]
                      |
                      +-- O*NET-SOC Code --- ESSENTIAL_SKILLS
                      +-- O*NET-SOC Code --- SOFTWARE_SKILLS
```

## Relationship Table

| Join Key | Left Table | Right Table | Relationship | Overlap | Why It Matters |
|----------|------------|-------------|--------------|---------|----------------|
| EmployeeID | employee_attrition | engagement | Disjoint | 0% | Different ID spaces (1-500 vs 3427+), likely different populations. Tables cannot be joined. |
| JobRole/Title | employee_attrition | occupation | No match | 0% | HR internal labels vs O*NET standard titles. Manual mapping needed to connect. |
| O*NET-SOC Code | occupation | essential_skills | 1:many | 100% | Clean foreign key. ~20 skills per occupation. Safe to join. |
| O*NET-SOC Code | occupation | software_skills | 1:many | 100% | Clean foreign key. Variable skills per occupation. Safe to join. |

## Key Findings

- Employee and engagement tables cannot be joined (0% overlap) — they use different ID systems
- Employee JobRole doesn't match occupation Title (different vocabularies) — would need manual mapping
- Occupation links cleanly to both skills tables via O*NET-SOC Code — this is the reliable join path

## Implications for ML Pipeline

- **Attrition model**: Trains on employee_attrition alone (cannot use engagement data)
- **Engagement analytics**: Runs on engagement table standalone (cannot link to employee records)
- **Skill-gap analysis**: Runs at occupation level only (cannot link to specific employees)
- **Day 2 feature engineering**: Focus on employee_attrition features; occupation-skills join is available for role-based features
