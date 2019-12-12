import os


def time_resolved_check(in_dir, container):
    """
    Compute over time results for many consecutive recordings.

    This was originally in burst_analysis.py.
    The idea was to look at the muscimol data over time and to compute
    burst statistics to check a change over time.

    However, it was removed from the main script as the burst_analysis.py
    script is stable, while this code is still in progress.

    """
    print("Computing time resolved tests")
    with open(
            os.path.join(in_dir, "nc_results", "bursttime.csv"), "w") as f:
        from collections import OrderedDict
        import operator
        info = OrderedDict()
        for i in range(container.get_num_data()):
            c_info = container.get_index_info(i)
            compare, tetrode = os.path.splitext(c_info["Spike"])
            r_type = compare.split("_")[-1]
            final_c = compare[:-(len(r_type) + 1)]
            add_info = (int(r_type), int(tetrode[1:]), c_info["Units"], i)
            if final_c[2:].startswith("11"):
                str_part = final_c[2:13]
            else:
                str_part = final_c[2:20]
            if str_part in info:
                info[str_part].append(add_info)
            else:
                info[str_part] = [add_info]
        for key, val in info.items():
            val = sorted(val, key=operator.itemgetter(0, 1))
            info[key] = val

        f.write("Name")
        for i in range(1, 18):
            f.write(",{}".format(i))
        f.write("\n")
        for key, val in info.items():
            t_units = OrderedDict()
            for s_val in val:
                r_type, t, units, idx = s_val
                if str(t) in t_units:
                    for u in units:
                        if u in t_units[str(t)]:
                            t_units[str(t)][u].append((r_type, idx))
                else:
                    t_units[str(t)] = OrderedDict()
                    for u in units:
                        t_units[str(t)][u] = [(r_type, idx)]

            for tetrode, units in t_units.items():
                for unit, idxs in units.items():
                    f.write(
                        key + "__" + str(tetrode) + "__" + str(unit) + ",")
                    burst_arr = np.full(17, np.nan)
                    for i in idxs:
                        unit_o_idx = container.get_units(i[1]).index(unit)
                        data = container.get_data_at(i[1], unit_o_idx)
                        data.burst()
                        p_burst = data.get_results()["Propensity to burst"]
                        burst_arr[i[0] - 1] = p_burst
                    o_str = ""
                    for b in burst_arr:
                        o_str = o_str + "{},".format(b)
                    f.write(o_str[:-1] + "\n")
