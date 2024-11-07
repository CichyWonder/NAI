import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input variables
"""
Implementacja:
    - Michał Cichowski s20695
    
Input variables:
- np.arange(0, 101, 1) defines a range of 0-100 (inclusive) with a step of 1.
"""
customer_importance = ctrl.Antecedent(np.arange(0, 101, 1), 'customer_importance')
ticket_severity = ctrl.Antecedent(np.arange(0, 101, 1), 'ticket_severity')
ticket_importance = ctrl.Antecedent(np.arange(0, 101, 1), 'ticket_importance')

# Output variable
ticket_priority = ctrl.Consequent(np.arange(0, 101, 1), 'ticket_priority')

# Define membership functions for input variables
"""
Membership functions for input variables:
- 'LOW', 'MEDIUM', and 'HIGH' represent linguistic labels.
- fuzz.trimf defines triangular membership functions.
"""
customer_importance['LOW'] = fuzz.trimf(customer_importance.universe, [0, 25, 50])
customer_importance['MEDIUM'] = fuzz.trimf(customer_importance.universe, [0, 50, 100])
customer_importance['HIGH'] = fuzz.trimf(customer_importance.universe, [50, 100, 100])

ticket_severity['LOW'] = fuzz.trimf(ticket_severity.universe, [0, 0, 30])
ticket_severity['MEDIUM'] = fuzz.trimf(ticket_severity.universe, [0, 30, 70])
ticket_severity['HIGH'] = fuzz.trimf(ticket_severity.universe, [30, 100, 100])

ticket_importance['LOW'] = fuzz.trimf(ticket_importance.universe, [0, 0, 30])
ticket_importance['MEDIUM'] = fuzz.trimf(ticket_importance.universe, [0, 30, 70])
ticket_importance['HIGH'] = fuzz.trimf(ticket_importance.universe, [30, 100, 100])

# Define membership functions for output variable
"""
Membership functions for the output variable 'ticket_priority':
- 'LOW', 'MEDIUM', and 'HIGH' represent linguistic labels.
- fuzz.trimf defines triangular membership functions.
"""
ticket_priority['LOW'] = fuzz.trimf(ticket_priority.universe, [0, 0, 40])
ticket_priority['MEDIUM'] = fuzz.trimf(ticket_priority.universe, [0, 40, 70])
ticket_priority['HIGH'] = fuzz.trimf(ticket_priority.universe, [40, 100, 100])

# Define the rules based on user requirements
"""
Fuzzy logic rules:
- Rule conditions are defined based on input and output linguistic labels and their relationships.
"""
rule1 = ctrl.Rule(customer_importance['LOW'] & ticket_severity['HIGH'] & ticket_importance['HIGH'], ticket_priority['MEDIUM'])
rule2 = ctrl.Rule(ticket_severity['LOW'] & ticket_importance['LOW'], ticket_priority['LOW'])
rule3 = ctrl.Rule((customer_importance['MEDIUM'] | customer_importance['HIGH']) &
                  (ticket_severity['MEDIUM'] | ticket_severity['HIGH']) &
                  ticket_importance['HIGH'], ticket_priority['HIGH'])
rule4 = ctrl.Rule(customer_importance['LOW'] &
                  (ticket_severity['MEDIUM'] | ticket_importance['HIGH']) &
                  (ticket_severity['HIGH'] | ticket_importance['HIGH']), ticket_priority['MEDIUM'])

# Create control system
ticket_priority_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
ticket_priority_simulation = ctrl.ControlSystemSimulation(ticket_priority_ctrl)

# User input and fuzzy logic computation
"""
User input:
- User provides values for 'customer_importance_input', 'ticket_severity_input', and 'ticket_importance_input'.
- The fuzzy logic system computes the 'ticket_priority_output' based on the user input.
- The 'priority_level' is determined based on the computed 'ticket_priority_output'.
"""
customer_importance_input = float(input("Enter customer importance (0-100): "))
ticket_severity_input = float(input("Enter ticket severity (0-100): "))
ticket_importance_input = float(input("Enter ticket importance (0-100): "))

# Pass user input values (normalized) to the ControlSystemSimulation and compute the output
ticket_priority_simulation.input['customer_importance'] = customer_importance_input
ticket_priority_simulation.input['ticket_severity'] = ticket_severity_input
ticket_priority_simulation.input['ticket_importance'] = ticket_importance_input

# Compute the result
ticket_priority_simulation.compute()

# Output
ticket_priority_output = ticket_priority_simulation.output['ticket_priority']
print("Ticket Priority:", ticket_priority_output)

# Determine priority level based on computed output
if ticket_priority_output <= 40:
    priority_level = 'LOW'
elif ticket_priority_output <= 70:
    priority_level = 'MEDIUM'
else:
    priority_level = 'HIGH'

print("Priority Level:", priority_level)

# Visualize the fuzzy logic system
ticket_priority.view(sim=ticket_priority_simulation)
