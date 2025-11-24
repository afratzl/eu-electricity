def plot_analysis(stats_data, source_type, output_file):
    """
    Create vertical plots - percentage on top, absolute below
    """
    if not stats_data:
        print("No data for plotting")
        return None

    print("\nCreating plots...")

    # VERTICAL LAYOUT: 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))

    colors = {
        'today': '#FF4444',
        'yesterday': '#FF8800',
        'week_ago': '#4444FF',
        'year_ago': '#44AA44',
        'two_years_ago': '#AA44AA',
        'today_projected': '#FF4444',
        'yesterday_projected': '#FF8800'
    }

    linestyles = {
        'today': '-',
        'yesterday': '-',
        'week_ago': '-',
        'year_ago': '-',
        'two_years_ago': '-',
        'today_projected': '--',
        'yesterday_projected': '--'
    }

    labels = {
        'today': 'Today',
        'yesterday': 'Yesterday',
        'week_ago': 'Previous Week (avg ± std)',
        'year_ago': 'Same Period Last Year (avg ± std)',
        'two_years_ago': 'Same Period 2 Years Ago (avg ± std)',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }

    time_labels = create_time_axis()
    source_name = DISPLAY_NAMES[source_type]
    
    # PLOT 1 (TOP): PERCENTAGE
    ax1.set_title(f'{source_name} as Percentage of Total EU Generation',
                  fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=22, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Percentage (%)', fontsize=22, fontweight='bold', labelpad=15)

    max_percentage = 0

    plot_order = ['week_ago', 'year_ago', 'two_years_ago', 'yesterday', 'today', 
                  'yesterday_projected', 'today_projected']

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'percentage_mean' not in data or len(data['percentage_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['percentage_mean']))
        y_values = data['percentage_mean'].copy()

        max_percentage = max(max_percentage, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax1.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=4.5, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'percentage_std' in data:
            std_values = data['percentage_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_percentage = max(max_percentage, np.nanmax(upper_bound))

            ax1.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlim(0, len(time_labels))

    if max_percentage > 0:
        ax1.set_ylim(0, max_percentage * 1.05)
    else:
        ax1.set_ylim(0, 50)

    # PLOT 2 (BOTTOM): ABSOLUTE VALUES
    ax2.set_title(f'EU {source_name} Energy Production',
                  fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=22, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Energy Production (MW)', fontsize=22, fontweight='bold', labelpad=15)

    max_energy = 0

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'energy_mean' not in data or len(data['energy_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['energy_mean']))
        y_values = data['energy_mean'].copy()

        max_energy = max(max_energy, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=4.5, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'energy_std' in data:
            std_values = data['energy_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_energy = max(max_energy, np.nanmax(upper_bound))

            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_xlim(0, len(time_labels))
    ax2.set_ylim(0, max_energy * 1.05)

    # X-axis ticks for both plots
    tick_positions = np.arange(0, len(time_labels), 8)
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)

    # Legend below plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=18, frameon=False, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()

    return output_file
