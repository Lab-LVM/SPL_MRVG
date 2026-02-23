import os 
import pickle
import numpy as np
import scipy.stats as st

def normalize_beat(strength, mean, std):
    min_val = np.min(strength)
    max_val = np.max(strength)
    normalized_strength = (strength - np.mean(strength)) / np.std(strength)
    return normalized_strength


def compute_metrics(music_times, visual_times, music_beats, visual_beats, window_size, penalty_weight):
    total_time_difference = 0
    total_strength_difference = 0
    total_difference = 0
    total_penalty = 0
    total_num_matched = 0
    matched_visual_indices = set()
    penalized_indices = set()  

    music_noise_min = np.quantile(music_beats, 0.2)
    noise_flag = music_beats > music_noise_min
    music_beats = music_beats[noise_flag]
    music_times = music_times[noise_flag]
    visual_beats = np.clip(visual_beats, a_min=-0.3, a_max=None)

    for i, music_time in enumerate(music_times):
        start = music_time - window_size
        end = music_time + window_size
        potential_matches = np.where((visual_times >= start) & (visual_times <= end))[0]
        potential_matches = [idx for idx in potential_matches if
                             idx not in matched_visual_indices and idx not in penalized_indices]

        if len(potential_matches) > 0:
            closest_idx = potential_matches[np.argmin(np.abs(visual_times[potential_matches] - music_time))]
            matched_visual_indices.add(closest_idx) 

            closest_visual_time = visual_times[closest_idx]
            time_difference = (np.abs(music_time - closest_visual_time))
            total_time_difference += time_difference

            strength_difference = (np.abs(music_beats[i] - visual_beats[closest_idx]))
            total_strength_difference += strength_difference
            total_num_matched += 1

            if len(potential_matches) > 1:
                # extra_matches = [idx for idx in potential_matches if idx != closest_idx]
                extra_matches = [idx for idx in potential_matches]
                extra_visual_beats = np.asarray([visual_beats[idx] for idx in extra_matches])
                extra_visual_time = np.asarray([visual_times[idx] for idx in extra_matches])
                extra_penalty = sum(np.abs(extra_visual_beats - music_beats[i])) + sum(
                    np.abs(extra_visual_time - music_times[i]))
                total_penalty += extra_penalty
                penalized_indices.update(extra_matches) 
        elif music_beats[i] > music_noise_min:
            total_penalty += (music_beats[i] - music_noise_min) * penalty_weight

    avg_time_difference = total_time_difference / len(music_times)
    avg_strength_difference = total_strength_difference / len(music_times)
    avg_difference = total_difference / len(music_times)

    return avg_time_difference, avg_strength_difference, avg_difference, total_penalty


def compute_final_similarity_score(avg_time_difference, avg_strength_difference, avg_difference, total_penalty,
                                   max_time_threshold=1.0, max_penalty_threshold=100, window_size=0.5, ratio=0.5):
    similarity_score = max(0, 1 - avg_difference)
    time_similarity_score = avg_time_difference
    strength_similarity_score = avg_strength_difference
    penalty_score = total_penalty / max_penalty_threshold

    final_score = ratio / 2 * (time_similarity_score + strength_similarity_score) + (1 - ratio) * penalty_score
    return final_score


def compute_metric(genres, directory_paths, music_name_list, result_dict, rule, user_response, threshold, window_size,
                   penalty_weight, ratio):
    metric_score = []
    music_mean = 5.016024589538574
    music_std = 3.7560012340545654
    visual_mean = 0.3050067803499724
    visual_std = 0.24879122097399567

    for i, genre in enumerate(genres):
        music_names = music_name_list[i]
        directory_path = directory_paths[i]
        for music_name in music_names:
            try:
                with open(os.path.join(directory_path, f"{music_name}_audio_beat.pkl"), "rb") as fr:
                    music_beats = pickle.load(fr)

                music_beats = normalize_beat(music_beats, music_mean, music_std)

                with open(os.path.join(directory_path, f"{music_name}_audio_time.pkl"), "rb") as fr:
                    music_times = pickle.load(fr)

                for method in ['diff_smoothing', 'cqt', 'cqt_diff', 'MA','raw_smoothing']:
                    with open(os.path.join(directory_path, f"{music_name}_{method}_beat.pkl"), "rb") as fr:
                        visual_beats = pickle.load(fr)

                    with open(os.path.join(directory_path, f"{music_name}_{method}_time.pkl"), "rb") as fr:
                        visual_times = pickle.load(fr)
                    music_beats = np.array(list(music_beats))
                    visual_beats = np.array(list(visual_beats))
                    visual_beats = normalize_beat(visual_beats, visual_mean, visual_std)

                    music_times = np.array(list(music_times))
                    visual_times = np.array(list(visual_times))

                    avg_time_difference, avg_strength_difference, difference, total_penalty = compute_metrics(
                        music_times, visual_times, music_beats, visual_beats, window_size, penalty_weight
                    )

                    final_score = compute_final_similarity_score(avg_time_difference, avg_strength_difference,
                                                                 difference,
                                                                 total_penalty, max_penalty_threshold=len(music_beats),
                                                                 window_size=window_size, ratio=ratio)
                    result_dict[genre][music_name][method] = final_score
                    metric_score.append(final_score)

            except Exception as e:
                print(f"Error processing {music_name}: {e}")


    order = [[2, 3, 0, 1], [3, 1, 2, 0], [3, 2, 0, 1]]
    flag = []
    avg_results = []
    var_results = []
    avg_results_flat = []
    avg_results_per_method = [0 for _ in range(5)]
    raw_total = [[] for _ in range(5)]
    for i, genre in enumerate(genres):
        avg_results.append([])
        var_results.append([])
        for j, method in enumerate(['cqt', 'MA', 'cqt_diff', 'diff_smoothing','raw_smoothing']):
            score = 0
            count = 0
            raw = []
            for m in music_name_list[i]:
                value = result_dict[genre][m][method] * 100
                score += value
                raw.append(value)
                raw_total[j].append(value)
                count += 1
            score /= count
            avg_results[i].append(round(float(score), 3))
            var = st.sem(raw) * st.t.ppf((1 + 0.95) / 2., len(raw) - 1)
            var_results[i].append(round(float(var), 3))
            avg_results_flat.append(score)
            avg_results_per_method[j] += score


    var_results.append([])
    for i in range(5):
        var = st.sem(raw_total[i]) * st.t.ppf((1 + 0.95) / 2., len(raw_total[i]) - 1)
        var_results[-1].append(var)
    corrcoef = None

    if True:
        return True, avg_results, var_results, corrcoef
    else:
        return False, avg_results, var_results, corrcoef
