import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def generate_visuals(file_path, output_folder="visuals"):
    df = pd.read_csv(file_path, encoding="ISO-8859-1", delimiter=",", on_bad_lines="skip", low_memory=False)

    # Normalizing factors
    min_year = 1753
    max_year = 2104

    # Reverse normalize year_of_construction to actual years
    df['actual_year_of_construction'] = min_year + (df['year_of_construction'] * (max_year - min_year))

    df_selected = df[['totalprimaryenergyfact', 'firstenerprodconvfactor', 'firstenerproddelivered', 
                  'secondenerproddelivered', 'thirdenerproddelivered']].copy()

    # Energy savings cost
    df_selected['energy_savings'] = df_selected[['firstenerproddelivered', 'secondenerproddelivered', 'thirdenerproddelivered']].sum(axis=1)
    df_selected['cost_per_unit'] = df_selected['firstenerprodconvfactor']

    # Cost savings per investment dollar
    df_selected['investment'] = df_selected['cost_per_unit']
    df_selected['cost_savings_per_dollar'] = df_selected['energy_savings'] / df_selected['investment']
    
    # ### ---Visualization 1: Which Counties Have the Worst CO₂ Emissions?
    # Group data by county and sum the total CO2 emissions
    county_emissions = df.groupby('countyname')['totalco2emissions'].sum().reset_index()

    # Sort by total CO2 emissions in descending order
    county_emissions = county_emissions.sort_values(by='totalco2emissions', ascending=False)

    # Plot the bar chart
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.barplot(x='totalco2emissions', y='countyname', data=county_emissions, palette='ocean')
    plt.title('Total CO2 Emissions by County', fontsize=16)
    plt.xlabel('Total CO2 Emissions', fontsize=12)
    plt.ylabel('County Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'countywise_distribution_of_co2_emissions.png'))
    
    # --- Visualization 2: Which Regions Have the Best Energy Ratings?

    # Plotting average BER rating by county
    county_berrating = df.groupby('countyname')['berrating'].mean().reset_index()
    county_berrating = county_berrating.sort_values(by='berrating', ascending=False)

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=county_berrating, x='berrating', y='countyname', palette='ocean')
    plt.title('Which Regions Have the Best Energy Ratings? 🏙️')
    plt.xlabel('Average BER Rating')
    plt.ylabel('County')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "top_geographical_regions_with_best_energy_ratings.png"))
    
    # --- Visualization 3: Where is the Most Energy Used for Heating?

    # Plotting total energy used for heating (primaryenergymainspace) by county
    county_heating_energy = df.groupby('countyname')['primaryenergymainspace'].sum().reset_index()
    county_heating_energy = county_heating_energy.sort_values(by='primaryenergymainspace', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=county_heating_energy, x='primaryenergymainspace', y='countyname', palette='ocean')
    plt.title('Where is the Most Energy Used for Heating? 🌡️')
    plt.xlabel('Total Energy Used for Heating (kWh)')
    plt.ylabel('County')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "most_common_energy_sources_used_for_heating.png"))
    
    # --- Visualization 4: Has Energy Efficiency Improved Over Time?

    # Normalizing factors
    min_year = 1753
    max_year = 2104

    # Reverse normalize year_of_construction to actual years
    df['actual_year_of_construction'] = min_year + (df['year_of_construction'] * (max_year - min_year))

    # Plotting the trend of energy used for heating (primaryenergymainspace) over year of construction
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='actual_year_of_construction', y='primaryenergymainspace', marker='o')
    plt.title('Has Energy Efficiency Improved Over Time? 📅')
    plt.xlabel('Year of Construction')
    plt.ylabel('Energy Used for Heating (kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "yearly_trend_in_energy_efficiency_improvements.png"))
        
    # --- Visualization 5: Have Heating-Related Emissions Reduced Over Time?

    # Plotting the trend of heating-related CO₂ emissions (co2mainspace) over year of construction
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='actual_year_of_construction', y='co2mainspace', marker='o')
    plt.title('Have Heating-Related Emissions Reduced Over Time? 🏡')
    plt.xlabel('Year of Construction')
    plt.ylabel('Heating-Related CO₂ Emissions (kgCO₂/kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "historical_trends_in_heating_emissions.png"))
    
    # --- Visualization 6: Do Better-Insulated Walls Reduce Emissions? 🏗️ --- ###

    # Plotting the relationship between wall insulation (U-value) and total CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='uvaluewall', y='totalco2emissions')
    plt.title('Do Better-Insulated Walls Reduce Emissions? 🏗️')
    plt.xlabel('Wall U-Value (Lower is Better Insulated)')
    plt.ylabel('Total CO₂ Emissions (kgCO₂/kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "impact_of_wall_insulation_on_co2_emissions.png"))
    
    # --- Visualization 7: How Much Heat is Lost Through the Roof? 

    # Plotting the relationship between roof insulation (U-value) and total CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='uvalueroof', y='totalco2emissions')
    plt.title('How Much Heat is Lost Through the Roof? 🏠')
    plt.xlabel('Roof U-Value (Lower is Better Insulated)')
    plt.ylabel('Total CO₂ Emissions (kgCO₂/kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "heat_loss_patterns_through_roofs_over_time.png"))
    
    # --- Visualization 8: Do Larger Walls Lead to More Energy Loss?

    # Plotting the relationship between wall area and total CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='wallarea', y='totalco2emissions')
    plt.title('Do Larger Walls Lead to More Energy Loss? 📐')
    plt.xlabel('Wall Area (sq m)')
    plt.ylabel('Total CO₂ Emissions (kgCO₂/kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "energy_loss_analysis_for_larger_walls.png"))
    
    # --- Visualization 9: Does Window Size Affect Energy Efficiency?

    # Plotting the relationship between window area and total CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='windowarea', y='totalco2emissions')
    plt.title('Does Window Size Affect Energy Efficiency? 🪟')
    plt.xlabel('Window Area (sq m)')
    plt.ylabel('Total CO₂ Emissions (kgCO₂/kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "relationship_between_window_size_and_energy_efficiency.png"))
    
    # --- Visualization 10: What Insulation Types Lead to Better BER Ratings?

    # Plotting the distribution of BER ratings for different insulation types
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='insulationtype', y='berrating', palette="ocean")
    plt.title('What Insulation Types Lead to Better BER Ratings? 🛠️')
    plt.xlabel('Insulation Type')
    plt.ylabel('Building Energy Rating (BER)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "analysis_of_insulation_types_vs_building_energy_ratings.png"))
    
    # --- Visualization 11: Do Buildings with Better Heat Recovery Use Less Energy?

    # Plotting the scatter plot for heat exchanger efficiency vs total CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='heatexchangereff', y='totalco2emissions', scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Do Buildings with Better Heat Recovery Use Less Energy? 🔄')
    plt.xlabel('Heat Exchanger Efficiency')
    plt.ylabel('Total CO2 Emissions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "comparison_of_heat_recovery_vs_energy_consumptio.png"))
    
    # --- Visualization 12: Do Larger Homes Emit More CO2?

    # Plotting the scatter plot with a regression line to show the relationship between ground floor area and CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='groundfloorarea(sq m)', y='totalco2emissions', scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Do Larger Homes Emit More CO2? 📏')
    plt.xlabel('Ground Floor Area (sq m)')
    plt.ylabel('Total CO2 Emissions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "carbon_emissions_from_larger_residential_homes.png"))
    
    # --- Visualization 13: Do Older Homes Have Worse Emissions?

    # Plotting the scatter plot with a regression line to show the relationship between year of construction and CO2 emissions
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='actual_year_of_construction', y='totalco2emissions', scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Do Older Homes Have Worse Emissions? 🏚️')
    plt.xlabel('Year of Construction')
    plt.ylabel('Total CO2 Emissions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "carbon_emissions_from_older_residential_homes.png"))
    
    # --- Visualization 14: Do Newer Homes Have Better Energy Ratings?

    # Plotting the scatter plot with a regression line to show the relationship between year of construction and BER rating
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='actual_year_of_construction', y='berrating', scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'blue'})
    plt.title('Do Newer Homes Have Better Energy Ratings? 🏠')
    plt.xlabel('Year of Construction')
    plt.ylabel('Building Energy Rating (BER)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "energy_efficiency_ratings_of_newer_homes.png"))
    
    # --- Visualization 15: Do Apartments Emit Less CO2 than Detached Houses?

    # Plotting the box plot to compare total CO2 emissions across different dwelling types
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dwellingtypedescr', y='totalco2emissions')
    plt.title('Do Apartments Emit Less CO2 than Detached Houses? 🏢🏡')
    plt.xlabel('Dwelling Type')
    plt.ylabel('Total CO2 Emissions (kg CO2)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "co2_emissions_comparison_apartments_vs_houses.png"))
    
    # --- Visualization 16: Do Taller Buildings Reduce Per-Unit CO2 Emissions?

    # Plotting the scatter plot with a regression line
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='nostoreys', y='totalco2emissions', scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Do Taller Buildings Reduce Per-Unit CO2 Emissions? 📊')
    plt.xlabel('Number of Storeys')
    plt.ylabel('Total CO2 Emissions (kg CO2)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "co2_emissions_analysis_for_taller_buildings.png"))
    
    # --- Visualization 17: Energy savings cost

    # Plotting the scatter plot with a regression line

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_selected, x='cost_per_unit', y='energy_savings', color='blue', s=100, alpha=0.6)
    plt.title("Energy Savings vs. Cost per Renovation Measure", fontsize=16)
    plt.xlabel("Cost per Unit (Energy Production Cost)", fontsize=12)
    plt.ylabel("Energy Savings (kWh/m²)", fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "correlation_between_energy_savings_and_costs.png"))
    
    # --- Visualization 18: Cost savings investment

    # Plotting the bar plot

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_selected, x='investment', y='cost_savings_per_dollar', palette='ocean')
    plt.title("Cost Savings per Investment Dollar", fontsize=16)
    plt.xlabel("Investment (Cost per Unit)", fontsize=12)
    plt.ylabel("Cost Savings per Investment Dollar", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "cost_savings_analysis_per_dollar_spent.png"))
        
    print("Visualizations saved in", output_folder)

if __name__ == "__main__":
    generate_visuals("data/processed_data.csv")