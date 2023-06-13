#!/usr/bin/env python
# coding: utf-8

# In[39]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


df = pd.read_csv('global-malaria-deaths-by-world-region.csv') #Datasets are found in Jupyter Notebook
df2 = pd.read_csv('malaria-deaths-by-age.csv')
df3 = pd.read_csv('malaria-deaths-by-region.csv')
df4 = pd.read_csv('malaria-death-rates.csv')
df5 = pd.read_csv('malaria-death-rates-by-age.csv')
df6 = pd.read_csv('incidence-of-malaria.csv')
df7 = pd.read_csv('malaria-prevalence-vs-gdp-per-capita.csv')
df8 = pd.read_csv('children-sleeping-under-treated-bednet.csv')


# In[46]:


tab1,tab2,tab3,tab4 = st.tabs(["Deaths","Death Rates","Incidence","Malaria V. Poverty"])


# In[38]:


with tab1:
    def generate_bar_plot(df):
        fig = px.bar(df, x='Entity', y='malaria_deaths', animation_frame='Year', range_y=[0, max(df['malaria_deaths'])], color='Entity',
                     title='Malaria Deaths per World Region',
                     labels={'malaria_deaths': 'Death Count', 'Year': 'Year', 'Entity': ''})

        fig.update_layout(xaxis={'categoryorder': 'total descending'})

        # Adjust animation settings
        fig.update_layout(updatemenus=[
            dict(type='buttons', showactive=False, buttons=[
                dict(label='Play', method='animate', args=[None, {
                    'frame': {'duration': 1000, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 500, 'easing': 'linear'}
                }]),
                dict(label='Stop', method='animate', args=[[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }])
            ])
        ])

        # Change animation effect to pop
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig.update_layout(transition=dict(duration=500, easing='linear'))

        # Add count annotations
        fig.update_traces(texttemplate='%{y}', textposition='outside')

        return fig

def main():
    st.title("Malaria Deaths per World Region")

    # Assuming you have a CSV file named 'global-malaria-deaths-by-world-region.csv' in the same directory
    df = pd.read_csv('global-malaria-deaths-by-world-region.csv')

    fig = generate_bar_plot(df)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()   
    
    def generate_pie_chart(df2):
        data_grouped = df2.groupby('Year').sum().reset_index()

        data_long = data_grouped.melt(id_vars='Year', value_vars=['Deaths - Malaria - Sex: Both - Age: 70+ years (Number)',
                                                                 'Deaths - Malaria - Sex: Both - Age: 50-69 years (Number)',
                                                                 'Deaths - Malaria - Sex: Both - Age: 15-49 years (Number)',
                                                                 'Deaths - Malaria - Sex: Both - Age: 5-14 years (Number)',
                                                                 'Deaths - Malaria - Sex: Both - Age: Under 5 (Number)'],
                                      var_name='Age Group', value_name='Death Count')

        fig = px.pie(data_long, values='Death Count', names='Age Group',
                     title='Malaria Deaths by Age Group',
                     labels={'Death Count': 'Death Count', 'Age Group': 'Age Group'})

        return fig

def main():
    st.title("Malaria Deaths by Age Group")

    # Assuming you have a CSV file named 'data.csv' in the same directory
    df2 = pd.read_csv('malaria-deaths-by-age.csv')

    fig = generate_pie_chart(df2)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_horizontal_bar_chart(df3):
        # Define the entities to exclude from the plot
        exclude_entities = ['Eastern Mediterranean Region (WHO)', 'East Asia & Pacific (WB)', 'African Region (WHO)',
                            'Europe & Central Asia (WB)', 'European Region (WHO)', 'G20',
                            'Latin America & Caribbean (WB)', 'Middle East & North Africa (WB)', 'North America (WB)',
                            'OECD Countries', 'Region of the Americas (WHO)', 'South-East Asia Region (WHO)',
                            'Sub-Saharan Africa (WB)', 'Western Pacific Region (WHO)', 'World', 'World Bank High Income',
                            'World Bank High Income', 'World Bank Lower Middle Income', 'World Bank Upper Middle Income',
                            'South Asia (WB)', 'World Bank Low Income']

        # Filter the dataframe to exclude the specified entities
        filtered_df = df3[~df3['Entity'].isin(exclude_entities)]

        # Group the filtered data by Entity and calculate the total death count for each entity
        df_grouped = filtered_df.groupby('Entity')['Deaths - Malaria - Sex: Both - Age: All Ages (Number)'].sum().reset_index()

        # Sort the data by the total death count in descending order
        df_sorted = df_grouped.sort_values(by='Deaths - Malaria - Sex: Both - Age: All Ages (Number)', ascending=False)

        # Select the top ten entities and sort in descending order
        top_ten = df_sorted.head(10).sort_values(by='Deaths - Malaria - Sex: Both - Age: All Ages (Number)',
                                                 ascending=True)

        # Create the horizontal bar chart
        fig = px.bar(top_ten, y='Entity', x='Deaths - Malaria - Sex: Both - Age: All Ages (Number)', orientation='h')

        # Update the layout
        fig.update_layout(
            title='Top 10 Countries With Highest Death by Malaria (2000-2019)',
            yaxis=dict(title='Country'),
            xaxis=dict(title='Death Count')
        )

        return fig

def main():
    st.title('Top 10 Countries With Highest Death by Malaria (2000-2019)')

    fig = generate_horizontal_bar_chart(df3)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_choropleth_map(df3):
        # Calculate the total deaths by country
        df_agg = df3.groupby(['Entity', 'Code'])['Deaths - Malaria - Sex: Both - Age: All Ages (Number)'].sum().reset_index()

        # Create the map visualization with Plotly
        fig = px.choropleth(df_agg, locations='Entity', locationmode='country names',
                            color='Deaths - Malaria - Sex: Both - Age: All Ages (Number)',
                            hover_name='Entity', projection='natural earth',
                            color_continuous_scale='YlOrRd', range_color=(0, 7000000))

        fig.update_layout(title='Malaria Deaths by Country',
                          coloraxis_colorbar=dict(title='Deaths'))

        return fig

def main():
    st.title('Malaria Deaths by Country')

    fig = generate_choropleth_map(df3)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_line_chart(df3):
        # Create traces for each entity
        data = []
        entities = df3['Entity'].unique()
        for i, entity in enumerate(entities):
            trace = go.Scatter(
                x=df3[df3['Entity'] == entity]['Year'],
                y=df3[df3['Entity'] == entity]['Deaths - Malaria - Sex: Both - Age: All Ages (Number)'],
                mode='lines',
                name=entity,
                visible=(i == 0)  # Set visibility for the first entity
            )
            data.append(trace)

        # Create a dropdown menu for entity selection
        buttons = []
        for entity in entities:
            button = dict(
                label=entity,
                method="update",
                args=[{"visible": [entity in trace['name'] for trace in data]},
                      {"title": f"Deaths from Malaria for {entity}"}]
            )
            buttons.append(button)

        # Create the layout with entity dropdown menu
        layout = go.Layout(
            title='Deaths from Malaria',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Number of Deaths'),
            updatemenus=[
                dict(
                    buttons=buttons,
                    active=0,
                    x=1.0,  # Set the x position to the right side
                    xanchor='right',  # Set the x anchor to right
                    y=1.0,  # Set the y position to the top
                    yanchor='top'  # Set the y anchor to top
                )
            ]
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        return fig

def main():
    st.title('Deaths from Malaria per Country')

    fig = generate_line_chart(df3)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()


# In[33]:


with tab2:
    def generate_choropleth_map(df4):
        # Create the map plot using Plotly Express
        fig = px.choropleth(
            data_frame=df4,
            locations='Code',
            locationmode='ISO-3',
            color='Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)',
            color_continuous_scale='YlOrRd',
            hover_name='Entity',
            hover_data={'Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)': ':,.2f'},
            labels={'Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)': 'Death Rate'},
            title='Death Rates - Malaria (Age-standardized)',
            animation_frame='Year'
        )

        # Configure layout
        fig.update_layout(
            geo=dict(
                showcoastlines=True,
                showframe=False,
                showocean=True,
                oceancolor='lightblue',
                showlakes=True,
                lakecolor='lightblue',
                projection_type='natural earth'
            ),
            coloraxis_colorbar=dict(
                title='Death Rate'
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        # Enable map zooming and panning
        fig.update_geos(
            resolution=50,
            showcountries=True,
            countrycolor='black',
            showsubunits=True,
            subunitcolor='gray'
        )

        return fig

def main():
    st.title('Death Rates - Malaria (Age-standardized)')

    fig = generate_choropleth_map(df4)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_bar_chart(df4):
        # Sort the dataset by the death rates in descending order for all years
        sorted_df = df4.sort_values(by='Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)', ascending=False)

        # Create a list of unique years in descending order
        years = sorted_df['Year'].unique()[::-1]

        # Create the figure with a dropdown selector for the year
        fig = go.Figure()

        # Add traces for each year
        for year in years:
            year_df = sorted_df[sorted_df['Year'] == year]
            top_10 = year_df.nlargest(10, 'Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)')  # Sort by highest rates

            fig.add_trace(go.Bar(
                x=top_10['Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)'][::-1],  # Reverse the order
                y=top_10['Entity'][::-1],  # Reverse the order
                orientation='h',
                name=str(year),
                visible=False  # Set initial visibility to False
            ))

        # Set visibility to True for the first year
        fig.data[0].visible = True

        # Create the dropdown selector
        dropdown_buttons = []
        for i, year in enumerate(years):
            dropdown_buttons.append(
                dict(
                    method='update',
                    args=[
                        {'visible': [i == j for j in range(len(years))]},  # Set visibility for each year
                        {'title': f'Top 10 Countries with Highest Death Rates by Malaria ({year})'}  # Update the title
                    ],
                    label=str(year)
                )
            )

        # Add the dropdown selector to the layout
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction='down',
                    showactive=True,
                    active=0,
                    x=0.5,  # Move the dropdown selector to the center
                    y=1.1,  # Move the dropdown selector slightly above the title
                    font=dict(color='black')
                )
            ],
            title='Top 10 Countries with Highest Death Rates by Malaria',
            xaxis_title='Death Rate',
            yaxis_title='Country',
        )

        return fig

def main():
    st.title('Top 10 Countries with Highest Death Rates by Malaria')

    fig = generate_bar_chart(df4)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_line_chart(df5):
        # Get unique values from the 'Entity' column of df5
        entities = df5['Entity'].unique().tolist()

        # Define age_groups with the relevant column names
        age_groups = [
            'Deaths - Malaria - Sex: Both - Age: Under 5 (Rate)',
            'Deaths - Malaria - Sex: Both - Age: 70+ years (Rate)',
            'Deaths - Malaria - Sex: Both - Age: 50-69 years (Rate)',
            'Deaths - Malaria - Sex: Both - Age: 15-49 years (Rate)',
            'Deaths - Malaria - Sex: Both - Age: 5-14 years (Rate)'
        ]

        # Remove specific columns from age_groups list
        age_groups = [col for col in age_groups if col not in ['Deaths - Malaria - Sex: Both - Age: All Ages (Rate)',
                                                               'Deaths - Malaria - Sex: Both - Age: Age-standardized (Rate)']]

        # Create traces for each age group for all entities
        data = []
        for age_group in age_groups:
            for entity in entities:
                trace = go.Scatter(
                    x=df5[df5['Entity'] == entity]['Year'],
                    y=df5[df5['Entity'] == entity][age_group],
                    mode='lines',
                    name=f'{entity} - {age_group.replace("Deaths - Malaria - Sex: Both - Age:", "Age:")}',
                    visible=entity == entities[0]  # Set visibility for the first entity
                )
                data.append(trace)

        # Create a dropdown menu for entity selection
        buttons = []
        for entity in entities:
            button = dict(
                label=entity,
                method="update",
                args=[{"visible": [entity in trace['name'] for trace in data]},
                      {"title": f"Death Rates per Age Group for {entity}"}]
            )
            buttons.append(button)

        # Create the layout with entity dropdown menu
        layout = go.Layout(
            title='Death Rates per Age Group',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Death Rate'),
            legend=dict(
                font=dict(
                    size=10
                )
            ),
            updatemenus=[dict(
                buttons=buttons,
                active=0,
                x=0.1,
                xanchor='left',
                y=1.0,
                yanchor='top'
            )]
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        return fig

def main():
    st.title('Death Rates per Age Group')

    fig = generate_line_chart(df5)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


# In[37]:


with tab3:
    def generate_choropleth_map(df6):
        # Calculate the maximum incidence value
        max_incidence = df6['Incidence of malaria (per 1,000 population at risk)'].max()

        # Create the map using Plotly Express
        fig = px.choropleth(df6,  # DataFrame containing the data
                            locations='Code',  # Column with country codes
                            locationmode='ISO-3',  # Set location mode to use country codes
                            color='Incidence of malaria (per 1,000 population at risk)',  # Column to use for color scale
                            hover_name='Entity',  # Column to use for hover labels
                            animation_frame='Year',  # Column to use for animation
                            color_continuous_scale='reds',  # Color scale
                            range_color=(0, max_incidence),  # Set the color scale range
                            title='Incidence of Malaria')  # Title of the chart

        # Update the layout for better interactivity
        fig.update_layout(geo=dict(showframe=False,  # Hide the frame around the map
                                   showcoastlines=False,  # Hide the coastlines
                                   projection_type='natural earth'),  # Use natural earth projection
                          coloraxis_colorbar=dict(title='Incidence',  # Set the colorbar title
                                                  lenmode='fraction',  # Set the length mode
                                                  len=0.75,  # Set the length
                                                  yanchor='middle',  # Set the y anchor
                                                  y=0.5,  # Set the y position
                                                  tickfont=dict(size=10)),  # Set the tick font size
                          autosize=True,  # Auto-size the plot
                          margin=dict(l=0, r=0, t=50, b=0),  # Adjust the margins
                          coloraxis_colorbar_x=-0.05,  # Adjust the colorbar x position
                          coloraxis_colorbar_len=0.6)  # Adjust the colorbar length

        return fig

def main():
    st.title('Incidence of Malaria')

    fig = generate_choropleth_map(df6)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_line_chart(df6):
        # Create traces for each entity
        data = []
        entities = df6['Entity'].unique()
        for i, entity in enumerate(entities):
            trace = go.Scatter(
                x=df6[df6['Entity'] == entity]['Year'],
                y=df6[df6['Entity'] == entity]['Incidence of malaria (per 1,000 population at risk)'],
                mode='lines',
                name=entity,
                visible=(i == 0)  # Set visibility for the first entity
            )
            data.append(trace)

        # Create a dropdown menu for entity selection
        buttons = []
        for entity in entities:
            button = dict(
                label=entity,
                method="update",
                args=[{"visible": [entity in trace['name'] for trace in data]},
                      {"title": f"Incidence of Malaria for {entity}"}]
            )
            buttons.append(button)

        # Create the layout with entity dropdown menu
        layout = go.Layout(
            title='Incidence of Malaria',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Incidence'),
            updatemenus=[
                dict(
                    buttons=buttons,
                    active=0,
                    x=1.0,  # Set the x position to the right side
                    xanchor='right',  # Set the x anchor to right
                    y=1.0,  # Set the y position to the top
                    yanchor='top'  # Set the y anchor to top
                )
            ]
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        return fig

def main():
    st.title('Incidence of Malaria per Country')

    fig = generate_line_chart(df6)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
    def generate_line_chart(df3):
        # Create traces for each entity
        data = []
        entities = df3['Entity'].unique()
        for i, entity in enumerate(entities):
            trace = go.Scatter(
                x=df3[df3['Entity'] == entity]['Year'],
                y=df3[df3['Entity'] == entity]['Deaths - Malaria - Sex: Both - Age: All Ages (Number)'],
                mode='lines',
                name=entity,
                visible=(i == 0)  # Set visibility for the first entity
            )
            data.append(trace)

        # Create a dropdown menu for entity selection
        buttons = []
        for entity in entities:
            button = dict(
                label=entity,
                method="update",
                args=[{"visible": [entity in trace['name'] for trace in data]},
                      {"title": f"Deaths from Malaria for {entity}"}]
            )
            buttons.append(button)

        # Create the layout with entity dropdown menu
        layout = go.Layout(
            title='Deaths from Malaria',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Number of Deaths'),
            updatemenus=[
                dict(
                    buttons=buttons,
                    active=0,
                    x=1.0,  # Set the x position to the right side
                    xanchor='right',  # Set the x anchor to right
                    y=1.0,  # Set the y position to the top
                    yanchor='top'  # Set the y anchor to top
                )
            ]
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        return fig

def main():
    st.title('Deaths from Malaria per Country')

    fig = generate_line_chart(df3)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    


# In[40]:


# Filter out rows with zero or negative population values
df7 = df7[df7['Population (historical estimates)'] > 0]

# Filter out rows with years below 1990
df7 = df7[df7['Year'] >= 1990]


# In[45]:


with tab4:
    # Apply logarithmic transformation to the axes
    df7['log_ShareInfected'] = df7['Current number of cases in the population per 100 people - Sex: Both - Age: Age-standardized - Cause: Malaria'].apply(lambda x: np.log10(x))
    df7['log_GDP'] = df7['GDP per capita, PPP (constant 2017 international $)'].apply(lambda x: np.log10(x))

    # Sort the unique years in ascending order
    sorted_years = sorted(df7['Year'].unique())

    # Create scatter plot using Plotly Express
    fig = px.scatter(
        df7,
        x='log_GDP',
        y='log_ShareInfected',
        hover_data=['Entity'],
        color=np.log10(df7['Population (historical estimates)']),
        color_continuous_scale='Viridis',
        size_max=10,
        animation_frame='Year',
        range_x=[df7['log_GDP'].min(), df7['log_GDP'].max()],
        range_y=[df7['log_ShareInfected'].min(), df7['log_ShareInfected'].max()]
    )

    fig.update_layout(
        title='Malaria Population Share vs. GDP',
        xaxis_title='GDP per capita, PPP (Log)',
        yaxis_title='Share of the Population with Malaria (Log)'
    )

    # Filter the scatter plot based on the first year
    fig.update_traces(visible=False)
    fig.data[0].visible = True

    # Display the scatter plot in Streamlit
    st.plotly_chart(fig)

    # Sort the DataFrame by the 'Year' column in ascending order
    df8 = df8.sort_values('Year')

    # Create a choropleth map
    fig = px.choropleth(
        df8,
        locations='Code',
        color='Use of insecticide-treated bed nets (% of under-5 population)',
        hover_name='Entity',
        animation_frame='Year',
        projection='natural earth',
        color_continuous_scale='Viridis',
        range_color=[0, 100]  # Set the range of the color scale
    )

    # Update map layout
    fig.update_layout(
        title='Use of Insecticide-Treated Bed Nets (% of under-5 population) by Country',
        geo=dict(
            showframe=True,  # Display the frame of the map
            showcoastlines=True,  # Display the coastlines
            projection_type='equirectangular',
            landcolor='lightgray',  # Set the color of the background/land
            coastlinecolor='darkgray',  # Set the color of the coastlines
            visible=False  # Hide the zooming effect when transitioning
        ),
        coloraxis_colorbar=dict(
            title='Use of ITN (%)',  # Set the colorbar title
            lenmode='fraction',  # Set the length mode of the colorbar
            len=0.25,  # Set the length of the colorbar relative to the plot
            xanchor='right',  # Set the horizontal anchor position of the colorbar
            x=0.95,  # Set the horizontal position of the colorbar
            yanchor='top',  # Set the vertical anchor position of the colorbar
            y=0.98,  # Set the vertical position of the colorbar
            bgcolor='rgba(255, 255, 255, 0.7)',  # Set the background color of the colorbar
            tickfont=dict(size=9)  # Set the font size of the colorbar ticks
        )
    )

    # Update map size
    fig.update_layout(height=500, margin={"r": 0, "l": 0, "b": 0, "t": 40})  # Set the height of the map and adjust the margins

    # Display the map in Streamlit
    st.plotly_chart(fig)

    # Define the list of countries
    countries = ['Sierra Leone', "Cote d'Ivoire", 'Liberia', 'Burkina Faso', 'Benin', 'Nigeria', 'Cameroon', 'Niger', 'Togo', 'Mozambique']

    # Filter the dataset for the selected countries
    filtered_df = df8[df8['Entity'].isin(countries)]

    # Create a line plot
    fig = px.line(filtered_df, x='Year', y='Use of insecticide-treated bed nets (% of under-5 population)', color='Entity')

    # Update plot layout
    fig.update_layout(
        title='Use of Insecticide-Treated Bed Nets (% of under-5 population)',
        xaxis_title='Year',
        yaxis_title='Percentage',
        legend_title='Country'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# In[ ]:




