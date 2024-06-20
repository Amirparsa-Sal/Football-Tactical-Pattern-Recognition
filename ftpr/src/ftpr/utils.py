def get_bin(x, y, bin_size=(4, 4)):
    return x // bin_size[0], y // bin_size[1]

def get_bin_center(bin_x, bin_y, bin_size=(4, 4)):
    return bin_x * bin_size[0] + bin_size[0] / 2, bin_y * bin_size[1] + bin_size[1] / 2

def get_interpolated_series(series, bin_size=(4, 4)):
	assert len(series) >= 2
	result = [series[0], series[1]]
	i = 1
	j = 2
	while j < len(series):
		this_bin = get_bin(result[i][0], result[i][1], bin_size=bin_size)
		prev_bin = get_bin(result[i-1][0], result[i-1][1], bin_size=bin_size)
		if abs(this_bin[0] - prev_bin[0]) > 1 or abs(this_bin[1] - prev_bin[1]) > 1:
			mid_point = [(result[i-1][0] + result[i][0]) / 2, (result[i-1][1] + result[i][1])/2]
			result.insert(i, mid_point)
		else:
			i += 1
			if i == len(result):
				result.append(series[j])
				j += 1
	return result
