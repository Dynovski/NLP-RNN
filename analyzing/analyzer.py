from plotly import graph_objs as go


class Analyzer:
    def __init__(self, data: 'pandas.DataFrame'):
        self.data = data

    def analyze_distribution(self, attribute: str, path: str = '') -> 'pandas.DataFrame':
        """
        Checks how many instances of given class of selected attributes is there in data.
        By specifying path, histogram is created and saved at provided location.

        :param attribute: str
            Name of the attribute for which to check distribution
        :param path: str
            Path to the file to save distribution plot
        :return: pandas.DataFrame
            data frame containing class names and number of instances for given attribute
        """
        assert attribute in self.data.columns

        distribution: 'pandas.DataFrame' = self.data.groupby(attribute)[attribute].agg('count')

        if path:
            fig = go.Figure()
            for column in distribution.columns:
                fig.add_trace(
                    go.Bar(
                        x=[column],
                        y=[distribution[column][0]],
                        name=column,
                        text=[distribution[column][0]],
                        textposition='auto'
                    )
                )
            fig.update_layout(
                title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
            )
            fig.write_image(path)

        return distribution
