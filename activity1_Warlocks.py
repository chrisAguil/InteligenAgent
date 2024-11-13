import agentpy as ap
import numpy as np
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt

# Base agent
class BaseWealthAgent(ap.Agent):
    """ Base agent with wealth """
    def setup(self):
        self.wealth = 5
        self.strategy_name = "Base"
        self.state_utility = 0
        self.cumulative_utility = 0

    def wealth_transfer(self):
        pass

# Only transfers if units > 2
class ConservativeAgent(BaseWealthAgent):
    def setup(self):
        super().setup()
        self.strategy_name = "Conservative"
        self.wealth = 5  
    
    def wealth_transfer(self):
        if self.wealth > 2:
            partner = list(self.model.agents.random(n=1))[0]
            partner.wealth += 1
            self.wealth -= 1

# only transfers to richer
class GreedyAgent(BaseWealthAgent):
    def setup(self):
        super().setup()
        self.strategy_name = "Greedy"
        self.wealth = 5
    
    def wealth_transfer(self):
        if self.wealth > 0:
            richer_partners = [agent for agent in self.model.agents if agent.wealth > self.wealth]
            if richer_partners:
                partner = random.choice(richer_partners)
                partner.wealth += 1
                self.wealth -= 1

# only to poorer
class CharitableAgent(BaseWealthAgent):
    def setup(self):
        super().setup()
        self.strategy_name = "Charitable"
        self.wealth = 5
    
    def wealth_transfer(self):
        if self.wealth > 0:
            poorest = min(self.model.agents, key=lambda x: x.wealth)
            if poorest.wealth < self.wealth:
                poorest.wealth += 1
                self.wealth -= 1

# multiple transfer if possible
class RiskTakingAgent(BaseWealthAgent):
    def setup(self):
        super().setup()
        self.strategy_name = "RiskTaker"
        self.wealth = 5  
    
    def wealth_transfer(self):
        if self.wealth > 2:
            partner = list(self.model.agents.random(n=1))[0]
            transfer = min(self.wealth - 1, 3)  
            partner.wealth += transfer
            self.wealth -= transfer

# Agent all-or-nothing
class AllOrNothingAgent(BaseWealthAgent):
    def setup(self):
        super().setup()
        self.strategy_name = "AllOrNothing"
        self.wealth = 5
    
    def wealth_transfer(self):
        if self.wealth > 0:
            partner = list(self.model.agents.random(n=1))[0]
            a = partner.wealth
            partner.wealth -= a
            self.wealth += a

# Gini function
def gini(x):
    """ Calculate Gini Coefficient """
    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean()  
    rmad = mad / np.mean(x)  
    return 0.5 * rmad

# Transfer Model
class WealthModel(ap.Model):
    """ A model of wealth transfers between different agent strategies """
    
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents['Base'], BaseWealthAgent) + \
                      ap.AgentList(self, self.p.agents['Conservative'], ConservativeAgent) + \
                      ap.AgentList(self, self.p.agents['Greedy'], GreedyAgent) + \
                      ap.AgentList(self, self.p.agents['Charitable'], CharitableAgent) + \
                      ap.AgentList(self, self.p.agents['RiskTaker'], RiskTakingAgent) + \
                      ap.AgentList(self, self.p.agents['AllOrNothing'], AllOrNothingAgent)
        
        # Initialize lists to store utility history
        self.state_utility_history = []
        self.cumulative_utility_history = []

        # Initialize a DataFrame to record wealth over time
        self.wealth_history = pd.DataFrame()
        self.step_count = 0  # Initialize step counter
        self.initial_wealth = [agent.wealth for agent in self.agents]  # Wealth at the beginning

    def step(self):
        self.agents.wealth_transfer()  # each agent does 1 transfer
        self.record_wealth()  # Record wealth after each step
        self.step_count += 1  # Increment the step count

        base_agents = [agent for agent in self.agents if agent.strategy_name == "Base"]
        state_utilities = [agent.state_utility for agent in base_agents]
        cumulative_utilities = [agent.cumulative_utility for agent in base_agents]

        self.state_utility_history.append(state_utilities)
        self.cumulative_utility_history.append(cumulative_utilities)

    def record_wealth(self):
        """ Record the wealth of each agent after each step. """
        wealths = [agent.wealth for agent in self.agents]
        types = [agent.strategy_name for agent in self.agents]
        step_data = pd.DataFrame({'Step': self.step_count, 'Wealth': wealths, 'Agent Type': types})
        self.wealth_history = pd.concat([self.wealth_history, step_data], ignore_index=True)

    def update(self):
        # Gini calculate
        wealths = [agent.wealth for agent in self.agents]
        self.record('Gini Coefficient', gini(wealths))

    def end(self):
        self.agents.record('wealth')
        self.evaluate_winners()  # Llama a la función que evalúa los ganadores

    def evaluate_winners(self):
        """ Evaluar quién ha ganado al final de la simulación """
        winners = []
        for i, agent in enumerate(self.agents):
            if agent.wealth > self.initial_wealth[i]:  # Condición de ganancia
                winners.append(agent.strategy_name)
        print(f"Ganan: {winners}")

# Hyper-parameters
parameters = {
    'agents': {
        'Base': 2,
        'Conservative': 2,
        'Greedy': 12,
        'Charitable': 2,
        'RiskTaker': 2,
        'AllOrNothing': 3
    },
    'steps': 100,
    'seed': 42,
}

# Run the model
model = WealthModel(parameters)
results = model.run()


#
steps = range(parameters['steps'])
stade_df = pd.DataFrame()
for agent_idx in range(len(model.state_utility_history[0])):
    agent_utility = [step_utility[agent_idx] for step_utility in model.state_utility_history]
    stade_df[f'Agent {agent_idx+1}'] = agent_utility
stade_df['Average'] = stade_df.mean(axis=1)

cumulative_df = pd.DataFrame()
for agent_idx in range(len(model.cumulative_utility_history[0])):
    agent_utility = [step_utility[agent_idx] for step_utility in model.cumulative_utility_history]
    cumulative_df[f'Agent {agent_idx+1}'] = agent_utility
cumulative_df['Average'] = cumulative_df.mean(axis=1)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot state-based utility
for column in stade_df.columns[:-1]:
    ax1.plot(steps, stade_df[column], alpha=0.3, color='gray', linewidth=1)

ax1.plot(steps, stade_df['Average'], color='red', linewidth=2, label='Average State Utility')
ax1.set_title('State-Based Utility Over Time (Base Agents)')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('State Utility\n(+1 for gain, -1 for loss)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot cumulative utility
for column in cumulative_df.columns[:-1]:
    ax2.plot(steps, cumulative_df[column], alpha=0.3, color='gray', linewidth=1)
ax2.plot(steps, cumulative_df['Average'], color='blue', linewidth=2, label='Average Cumulative Utility')
ax2.set_title('Cumulative Utility Over Time (Base Agents)')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Cumulative Utility\n(Total Wealth)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add final statistics
final_state_mean = stade_df['Average'].iloc[-1]
final_cumulative_mean = cumulative_df['Average'].iloc[-1]

stats_text = f'Final Statistics:\n'
stats_text += f'Avg State Utility: {final_state_mean:.2f}\n'
stats_text += f'Avg Cumulative Utility: {final_cumulative_mean:.2f}'

plt.figtext(0.02, 0.02, stats_text,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('base_agents_utility.png')
plt.show()






#


# Plot wealth history over time
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=model.wealth_history, x='Step', y='Wealth', hue='Agent Type', marker='o')

# plt.title('Historia de la Riqueza por Tipo de Agente')
# plt.xlabel('Paso')
# plt.ylabel('Riqueza')
# plt.legend(title='Tipo de Agente')
# plt.show()

# # Histogram of wealth distribution by agent type after the simulation
# agent_data = [(agent.strategy_name, agent.wealth) for agent in model.agents]
# df = pd.DataFrame(agent_data, columns=["Agent Type", "Wealth"])

# # Create a histogram of the wealth distribution for each agent type
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x="Wealth", hue="Agent Type", multiple="dodge", binwidth=1)
# plt.title("Distribución de la Riqueza por Tipo de Agente")
# plt.xlabel("Riqueza")
# plt.ylabel("Número de Agentes")
# plt.show()

# Print summary statistics
print("\nFinal Utility Statistics for Greedy Agents:")
print(f"Average State Utility: {final_state_mean:.2f}")
print(f"Average Cumulative Utility: {final_cumulative_mean:.2f}")