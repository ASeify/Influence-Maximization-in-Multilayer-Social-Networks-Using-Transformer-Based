import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

class Custom_plots:
    
    @staticmethod
    def plot_multiple_line(lines, label, color, marker):
        for line in lines:
            plt.plot(k_values, results_infection, label=label, linestyle='solid', marker=marker,  color=color)

    @staticmethod
    def plot_line(plt, infection_scale, k_values, method_results, label, color, marker):
        results_infection = []
        for k in k_values:
            results_infection.append(method_results[str(k)][infection_scale])
        plt.plot(k_values, results_infection, label=label, linestyle='solid', marker=marker,  color=color)

    @staticmethod
    def plot_lines(plt,compartion_parameter, infection_scale, k_values, method_results,
                    method_results_annalise, color, marker, visable_infection:bool=False):
        results_degree_average = []
        results_infection = []
        for k in k_values:
            results_degree_average.append(method_results_annalise[str(k)][compartion_parameter])
            results_infection.append(method_results[str(k)][infection_scale])
        plt.plot(k_values, results_degree_average, linestyle='dashed', marker=marker,  color=color)
        if visable_infection:
            plt.plot(k_values, results_infection, linestyle='solid', marker=marker,  color=color)

    @staticmethod
    def plot_comparison(compartion_parameter,infection_scale, k_values,
                        compartion_label, infection_label,
                        x_axis_lable, diagram_title,
                        list_of_results,
                        list_of_titles,
                        list_of_results_annalise,
                        plt_color_list,
                        plt_marker_list,
                        save_result_path,
                        legend_loc:str='best', legend_size='small', legend_framealpha = 0.5,
                        visable_infection:bool=False):
        # fig = plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xticks(k_values)
        handles, labels = plt.gca().get_legend_handles_labels()
        i = 0
        for j, results in enumerate(list_of_results):
            if i >= len(plt_color_list) - 1:
                i = 0
            plot_lines(plt,compartion_parameter,infection_scale, k_values, results,
                        list_of_results_annalise[j], plt_color_list[list_of_titles[j]],plt_marker_list[list_of_titles[j]], visable_infection)
            handles.append(Line2D([0], [0], label=list_of_titles[j], color=plt_color_list[list_of_titles[j]], linestyle='', marker=plt_marker_list[list_of_titles[j]]))
            i += 1

        if visable_infection:
            handles.append(Line2D([0], [0], label=infection_label, color='black', linestyle='solid'))
            handles.append(Line2D([0], [0], label=compartion_label, color='black', linestyle='dashed'))

        plt.rcParams['font.family'] = 'serif'  # Or 'sans-serif', 'monospace', etc.
        plt.rcParams['font.serif'] = ['Times New Roman']  # Try other fonts too
        
        plt.xlabel(x_axis_lable, fontsize=16)
        plt.ylabel(compartion_label, fontsize=16)
        # plt.title(diagram_title)
        plt.legend(handles=handles, loc=legend_loc, fontsize=legend_size, framealpha=legend_framealpha)

        # Tick labels
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # plt.margins(0,0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        plt.show()
        fig.savefig(save_result_path + diagram_title + '.png', bbox_inches='tight', pad_inches = 0.05)

    @staticmethod
    def plot_infection_comparison(infection_scale, k_values,
                                infection_label, x_axis_lable, diagram_title,
                                list_of_results, list_of_titles,
                                plt_color_list, plt_marker_list,
                                save_result_path,
                                legend_loc='best',
                                legend_size=14, legend_framealpha=0.5):
        # fig = plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xticks(k_values)

        
        
        for j, results in enumerate(list_of_results):
            plot_line(plt,infection_scale, k_values, results, list_of_titles[j],
                    plt_color_list[list_of_titles[j]], plt_marker_list[list_of_titles[j]])

        plt.rcParams['font.family'] = 'serif'  # Or 'sans-serif', 'monospace', etc.
        plt.rcParams['font.serif'] = ['Times New Roman']  # Try other fonts too
        
        # plt.subplots_adjust(bottom=.2, left=.2)
        plt.xlabel(x_axis_lable, fontsize=16)
        plt.ylabel(infection_label, fontsize=16)
        # plt.title(diagram_title, fontsize=20)
        plt.legend(loc=legend_loc, fontsize=legend_size, framealpha=legend_framealpha)

        # Tick labels
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # plt.margins(0,0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.show()
        fig.savefig(save_result_path + diagram_title + '.png', bbox_inches='tight', pad_inches = 0.05)

    @staticmethod
    def plot_lines_and_bars(
        x,    # List of x-values (e.g., k values)
        line_series:dict,    # List of dicts: [{'y': [...], 'label': 'Line 1', 'color': 'blue'}, ...]
        bar_series:dict,     # List of dicts: [{'y': [...], 'label': 'Bar 1', 'color': 'orange'}, ...]
        xlabel:str,
        ylabel_l:str,
        ylabel_r:str,
        rcParams:dict={'axes.titlesize': 16, 'axes.labelsize': 16,
                    'xtick.labelsize': 16, 'ytick.labelsize': 16,
                    'legend.loc':'upper left'},  # Optional rcParams for customization
        diagram_title:str='',
        save_result_path:str='',
    ):

        plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # or another font
        'axes.titlesize': rcParams.get('axes.titlesize', 16),
        'axes.labelsize': rcParams.get('axes.labelsize', 16),
        'xtick.labelsize': rcParams.get('xtick.labelsize', 16),
        'ytick.labelsize': rcParams.get('ytick.labelsize', 16),
    })

        x = np.array(x)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        bar_width = 0.8 / max(len(bar_series), 1)  # auto adjust bar width


        # Plot lines on top
        if line_series:
            for i, line in enumerate(line_series):
                ax1.plot(x, line['y'],
                        marker=line.get('marker', '.'),
                        color=line.get('color', 'black'),
                        label=line.get('label', f'Line {i+1}'),
                        zorder=2)


        # Plot bars first (behind lines)
        if bar_series:
            for i, bar in enumerate(bar_series):
                shift = (i - (len(bar_series) - 1) / 2) * bar_width
                ax2.bar(x + shift, bar['y'], width=bar_width,
                        color=bar.get('color', 'gray'),
                        alpha=0.3,
                        label=bar.get('label', f'Bar {i+1}'),
                        zorder=1)

        
        # Labels and ticks
        ax1.set_xlabel(xlabel, color='black')
        ax1.set_ylabel(ylabel_l, color='black')
        ax2.set_ylabel(ylabel_r, color='black')

        ax1.set_xticks(x)
        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Combine legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1, loc=rcParams.get('legend.loc', 'upper left'),
                fontsize=rcParams.get('legend.fontsize', 14),
                bbox_to_anchor=rcParams.get('legend.bbox_to_anchor', None),
                framealpha = rcParams.get('legend.framealpha', 0.5)
                )

        # plt.title("Multi Line and Bar Chart")
        fig.tight_layout()
        plt.show()
        fig.savefig(save_result_path + diagram_title + '.png', bbox_inches='tight', pad_inches = 0.05)

    @staticmethod
    def plot_infection_comparison_line_bar(k_values:list, list_of_results:dict,
                                        list_of_titles:dict, plt_color_list_dict:dict,
                                        plt_marker_list_dict:dict, compares_path:str,
                                        rcParams:dict={'axes.titlesize': 16, 'axes.labelsize': 16,
                                                            'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                                            'legend.loc':'upper left'}):
        line_series = []
        bar_series = []
        for j, results in enumerate(list_of_results):
            line_series.append({})
            bar_series.append({})

            line_series[-1]['y'] = []
            bar_series[-1]['y'] = []
            
            bar_series[-1]['label'] = list_of_titles[j]
            line_series[-1]['label'] = list_of_titles[j]

            line_series[-1]['color'] = plt_color_list_dict[list_of_titles[j]]
            bar_series[-1]['color'] = plt_color_list_dict[list_of_titles[j]]

            line_series[-1]['marker'] = plt_marker_list_dict[list_of_titles[j]]
            bar_series[-1]['marker'] = plt_marker_list_dict[list_of_titles[j]]

            for k in k_values:
                line_series[-1]['y'].append(results[str(k)]['infection'])
                bar_series[-1]['y'].append(results[str(k)]['percentage'])


        plot_lines_and_bars(
            x=k_values,
            line_series=line_series,
            bar_series=bar_series,
            xlabel='Seedset Size (k)',
            ylabel_l='Infection Node Count',
            ylabel_r='Percentage of Infected Nodes',
            rcParams = rcParams,
            diagram_title='Infection_Percentage_Comparison',
            save_result_path=compares_path
        )
            

        pass

