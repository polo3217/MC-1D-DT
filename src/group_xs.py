import openmc
import pandas as pd

def get_group_xs(filename = None, verbose = False ) -> pd.DataFrame:
    if filename is None:
        raise ValueError("Filename must be provided to read the statepoint file.")
    sp = openmc.StatePoint(filename)
    # ── 2. Pull XS values directly from the tally ────────────────────────────────
    t = sp.get_tally(name='reaction_rates')   # name you gave it in the OpenMC script
    df_one_group = t.get_pandas_dataframe()
    if verbose:
        print("=== Reaction Rate Tally DataFrame ===")
        print(df_one_group)

    t_flux = sp.get_tally(name='flux')
    df_flux = t_flux.get_pandas_dataframe()
    if verbose:
        print("\n=== Flux Tally DataFrame ===")
        print(df_flux)


    # I want to get the absorption, scatter, fiision and total in the same line for each cell and energy group. 
    # I can pivot the dataframe to achieve this.
    df_pivot_one_group = df_one_group.pivot_table(
        index=['cell', 'energy low [eV]', 'energy high [eV]'],
        columns='score',
        values='mean'
    ).reset_index()
    if verbose:
        print("\n=== Cleaned Reaction Rate DataFrame ===")
        print(df_pivot_one_group)

    df_cross_section = df_pivot_one_group.copy()

    for cell in df_cross_section['cell'].unique():
        mask = df_cross_section['cell'] == cell
        flux = df_flux.loc[df_flux['cell'] == cell, 'mean'].values

        df_cross_section.loc[mask, 'total'] = (df_cross_section.loc[mask, 'total'] / flux)
        df_cross_section.loc[mask, 'absorption'] = (df_cross_section.loc[mask, 'absorption'] / flux)
        df_cross_section.loc[mask, 'scatter'] = (df_cross_section.loc[mask, 'scatter'] / flux)
        df_cross_section.loc[mask, 'fission'] = (df_cross_section.loc[mask, 'fission'] / flux)

        

    if verbose:
        print("\n=== Final Cross Section DataFrame ===")
        print(df_cross_section)
    
    return df_cross_section
