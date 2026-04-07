import numpy as np
import pandas as pd
from scipy.signal import welch, stft, detrend
import neurokit2 as nk
import pywt
import antropy


# ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ

def compute_hrv_features(signal, fs):
    """Улучшенные HRV признаки с обработкой коротких записей"""
    try:
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

        if len(rpeaks['ECG_R_Peaks']) < 4:
            return {k: np.nan for k in ['meanRR','sdNN','RMSSD','pNN50','LF','HF','LF_HF']}

        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)

        if len(rpeaks['ECG_R_Peaks']) >= 10:
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
        else:
            hrv_freq = pd.DataFrame()

        def safe_val(df, key):
            return df[key].values[0] if key in df.columns and len(df) > 0 else np.nan

        return {
            'meanRR': safe_val(hrv_time, 'HRV_MeanNN'),
            'sdNN': safe_val(hrv_time, 'HRV_SDNN'),
            'RMSSD': safe_val(hrv_time, 'HRV_RMSSD'),
            'pNN50': safe_val(hrv_time, 'HRV_pNN50'),
            'LF': safe_val(hrv_freq, 'HRV_LF'),
            'HF': safe_val(hrv_freq, 'HRV_HF'),
            'LF_HF': safe_val(hrv_freq, 'HRV_LFHF')
        }
    except Exception as e:
        return {k: np.nan for k in ['meanRR','sdNN','RMSSD','pNN50','LF','HF','LF_HF']}


def create_nan_morphology_features():
    """Создает все морфологические признаки с NaN"""
    base_features = ['P_amp', 'Q_amp', 'R_amp', 'S_amp', 'T_amp',
                    'QRS_dur', 'QT_int', 'PR_int', 'ST_segment',
                    'ST_slope', 'J_point_amp', 'ST60_amp',
                    'PT_ratio', 'QR_ratio']
    return {feature: np.nan for feature in base_features}


def try_multiple_delineate_methods(signal, rpeaks, fs):
    """Пробует разные методы delineate"""
    methods = ["dwt", "cwt", "peak"]
    for method in methods:
        try:
            _, delineate_info = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method=method, show=False)
            if (len(delineate_info.get('ECG_P_Peaks', [])) > 0 or
                len(delineate_info.get('ECG_T_Peaks', [])) > 0):
                return delineate_info
        except:
            continue
    return None


def extract_detailed_morphology(signal, rpeaks, delineate_info, fs):
    """Извлечение детальных морфологических признаков когда delineate работает"""
    features = {}
    wave_types = ['P', 'Q', 'R', 'S', 'T']
    for wave in wave_types:
        peaks = delineate_info.get(f'ECG_{wave}_Peaks', [])
        if len(peaks) > 0:
            features[f'{wave}_amp'] = np.nanmean(signal[peaks])
        else:
            features[f'{wave}_amp'] = np.nan

    try:
        intervals = nk.ecg_interval(signal, rpeaks, sampling_rate=fs)
        if 'ECG_Q_Durations' in intervals and 'ECG_S_Durations' in intervals:
            q_durations = intervals['ECG_Q_Durations']
            s_durations = intervals['ECG_S_Durations']
            if len(q_durations) > 0 and len(s_durations) > 0:
                features['QRS_dur'] = np.nanmean(q_durations + s_durations)

        if 'ECG_T_Durations' in intervals:
            qt_intervals = intervals['ECG_T_Durations']
            if len(qt_intervals) > 0:
                features['QT_int'] = np.nanmean(qt_intervals)

        if 'ECG_P_Durations' in intervals:
            pr_intervals = intervals['ECG_P_Durations']
            if len(pr_intervals) > 0:
                features['PR_int'] = np.nanmean(pr_intervals)
    except:
        pass

    if not np.isnan(features.get('P_amp', np.nan)) and not np.isnan(features.get('T_amp', np.nan)):
        features['PT_ratio'] = features['P_amp'] / (features['T_amp'] + 1e-6)

    if not np.isnan(features.get('Q_amp', np.nan)) and not np.isnan(features.get('R_amp', np.nan)):
        features['QR_ratio'] = abs(features['Q_amp']) / (features['R_amp'] + 1e-6)

    return features


def extract_basic_morphology(signal, r_peaks, fs):
    """Упрощенный анализ морфологии когда delineate не работает"""
    features = {
        'P_amp': np.nan, 'Q_amp': np.nan, 'S_amp': np.nan, 'T_amp': np.nan,
        'QRS_dur': np.nan, 'QT_int': np.nan, 'PR_int': np.nan,
        'PT_ratio': np.nan, 'QR_ratio': np.nan
    }

    try:
        qrs_durations = []
        for r_peak in r_peaks:
            if r_peak > 50 and r_peak < len(signal) - 50:
                segment = signal[r_peak-25:r_peak+25]
                diff_segment = np.gradient(segment)
                threshold = 0.1 * np.max(np.abs(diff_segment))
                qrs_points = np.where(np.abs(diff_segment) > threshold)[0]
                if len(qrs_points) > 0:
                    qrs_dur = (qrs_points[-1] - qrs_points[0]) / fs * 1000
                    qrs_durations.append(qrs_dur)

        if qrs_durations:
            features['QRS_dur'] = np.nanmean(qrs_durations)
    except:
        pass

    return features


def simple_st_analysis(signal, r_peaks, fs):
    """Упрощенный анализ ST сегмента"""
    try:
        st_segments = []
        st_slopes = []

        for r_peak in r_peaks:
            if r_peak < len(signal) - int(0.2 * fs):
                j_point = min(r_peak + int(0.04 * fs), len(signal) - 1)
                st60_point = min(j_point + int(0.06 * fs), len(signal) - 1)

                j_amp = signal[j_point]
                st60_amp = signal[st60_point]

                st_segment = st60_amp - j_amp
                st_segments.append(st_segment)

                if st60_point > j_point:
                    st_slope = (st60_amp - j_amp) / 0.06
                    st_slopes.append(st_slope)

        return {
            'ST_segment': np.nanmean(st_segments) if st_segments else np.nan,
            'ST_slope': np.nanmean(st_slopes) if st_slopes else np.nan,
            'J_point_amp': np.nanmean([signal[min(r + int(0.04 * fs), len(signal)-1)] for r in r_peaks]) if r_peaks else np.nan,
            'ST60_amp': np.nanmean([signal[min(r + int(0.1 * fs), len(signal)-1)] for r in r_peaks]) if r_peaks else np.nan
        }
    except:
        return {
            'ST_segment': np.nan, 'ST_slope': np.nan,
            'J_point_amp': np.nan, 'ST60_amp': np.nan
        }


def compute_robust_morphological_features(signal, fs):
    """НАДЕЖНОЕ извлечение морфологических признаков с альтернативными методами"""
    try:
        features = {}
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)

        if len(rpeaks['ECG_R_Peaks']) < 2:
            return create_nan_morphology_features()

        r_peaks = rpeaks['ECG_R_Peaks']
        features['R_amp'] = np.nanmean(cleaned[r_peaks])

        delineate_info = try_multiple_delineate_methods(cleaned, rpeaks, fs)

        if delineate_info:
            features.update(extract_detailed_morphology(cleaned, rpeaks, delineate_info, fs))
        else:
            features.update(extract_basic_morphology(cleaned, r_peaks, fs))

        features.update(simple_st_analysis(cleaned, r_peaks, fs))
        return features

    except Exception as e:
        return create_nan_morphology_features()


def compute_frequency_features(signal, fs):
    """Частотные признаки на основе PSD и STFT"""
    try:
        f, Pxx = welch(signal, fs=fs, nperseg=min(fs*2, len(signal)//2))
        f_mean = np.sum(f * Pxx) / np.sum(Pxx)
        f_std = np.sqrt(np.sum(((f - f_mean)**2) * Pxx) / np.sum(Pxx))

        _, _, Zxx = stft(signal, fs=fs, nperseg=min(256, len(signal)//4))
        stft_energy = np.mean(np.abs(Zxx)**2)

        return {
            'PSD_mean_freq': f_mean,
            'PSD_std_freq': f_std,
            'STFT_energy': stft_energy,
            'PSD_power_mean': np.mean(Pxx),
            'PSD_power_max': np.max(Pxx)
        }
    except Exception:
        return {k: np.nan for k in ['PSD_mean_freq','PSD_std_freq','STFT_energy','PSD_power_mean','PSD_power_max']}


def compute_wavelet_features(signal):
    """Вейвлет признаки (энергии уровней)"""
    try:
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        energies = {f'wavelet_E{i+1}': np.sum(np.square(c)) for i, c in enumerate(coeffs)}
        entropy = antropy.spectral_entropy(signal, sf=500, method='welch')
        return {**energies, 'wavelet_entropy': entropy}
    except Exception:
        return {'wavelet_E1': np.nan, 'wavelet_E2': np.nan, 'wavelet_E3': np.nan, 'wavelet_E4': np.nan, 'wavelet_entropy': np.nan}


def compute_signal_quality(signal):
    """Метрики качества сигнала"""
    try:
        signal_det = detrend(signal)
        noise_power = np.mean((signal - signal_det)**2)
        if noise_power > 0:
            snr = 10 * np.log10(np.mean(signal**2) / noise_power)
        else:
            snr = np.nan
        prop_censored = np.mean(np.abs(signal) > 5*np.std(signal))
        return {'SNR_dB': snr, 'prop_censored': prop_censored}
    except Exception:
        return {'SNR_dB': np.nan, 'prop_censored': np.nan}


def compute_myocardial_features(signal, fs):
    """Признаки для диагностики инфаркта миокарда"""
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)

        features = {
            'Q_wave_amp': np.nan,
            'Q_wave_dur': np.nan,
            'pathological_Q': False
        }

        q_amplitudes = []
        for r_peak in rpeaks['ECG_R_Peaks']:
            if r_peak > 20:
                search_start = max(0, r_peak - int(0.08 * fs))
                search_end = r_peak
                segment = cleaned[search_start:search_end]
                if len(segment) > 0:
                    min_idx = np.argmin(segment)
                    q_amplitude = segment[min_idx]
                    if q_amplitude < -0.05:
                        q_amplitudes.append(abs(q_amplitude))

        if q_amplitudes:
            features.update({
                'Q_wave_amp': np.nanmean(q_amplitudes),
                'Q_wave_dur': np.nan,
                'pathological_Q': np.nanmean(q_amplitudes) > 0.1
            })

        return features

    except Exception:
        return {
            'Q_wave_amp': np.nan,
            'Q_wave_dur': np.nan,
            'pathological_Q': False
        }


def extract_features_from_ecg_record(ecg_record, fs):
    """Объединение всех улучшенных признаков"""
    features = {}

    for lead_name, signal in ecg_record.items():
        lead_features = {}

        lead_features.update(compute_hrv_features(signal, fs))
        lead_features.update(compute_robust_morphological_features(signal, fs))
        lead_features.update(compute_frequency_features(signal, fs))
        lead_features.update(compute_wavelet_features(signal))
        lead_features.update(compute_signal_quality(signal))

        if lead_name in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            lead_features.update(compute_myocardial_features(signal, fs))

        features.update({f"{lead_name}_{k}": v for k, v in lead_features.items()})

    return features


# ФУНКЦИИ ПРЕДОБРАБОТКИ ДАННЫХ

def categorize_heart_axis(value):
    """Категоризует отклонение электрической оси сердца по значимости"""
    critical_values = {'LAD', 'ALAD', 'RAD', 'ARAD'}
    medium_values = {'AXR', 'AXL'}
    normal_values = {'MID'}
    technical_values = {'AXIND', 'AXSUP', 'AXPOS', 'AXVER', 'AXHOR', 'TRSLT', 'TRSRT', 'CCWRT', 'CWRT'}

    if pd.isna(value) or value == 'NaN':
        return np.nan
    elif value in critical_values:
        return 'critical'
    elif value in medium_values:
        return 'medium'
    elif value in normal_values:
        return 'normal'
    elif value in technical_values:
        if value == 'AXIND':
            return 'medium'
        else:
            return 'Unknown'
    else:
        return 'Unknown'


def create_col_list_from_search_string(df, *search_strings):
    """Создает список признаков по нескольким строкам поиска в названии"""
    found_columns = []
    for search_string in search_strings:
        columns_for_string = [col for col in df.columns if search_string in col]
        found_columns.extend(columns_for_string)
    found_columns = list(set(found_columns))
    return found_columns


# Словарь с медианными значениями (из ноутбука)
REPLACEMENT_VALUES = {
    'age': 61.0,
    'I_prop_censored': 0.011,
    'II_wavelet_entropy': 5.519567349041058,
    'II_SNR_dB': 21.942648523132625,
    'III_PSD_std_freq': 7.550265037500206,
    'III_wavelet_E2': 1.5620259888461931,
    'AVR_RMSSD': 58.73670062235365,
    'AVR_STFT_energy': 8.445397813722286e-05,
    'AVR_SNR_dB': 22.375347144873952,
    'AVR_prop_censored': 0.011,
    'AVL_STFT_energy': 5.60859315034384e-05,
    'AVL_SNR_dB': 19.340258008946805,
    'AVF_SNR_dB': 20.149029495307428,
    'V1_RMSSD': 29.05932629027116,
    'V1_STFT_energy': 0.0001394457129169,
    'V1_wavelet_entropy': 5.424061319198822,
    'V1_SNR_dB': 23.659160937764973,
    'V2_STFT_energy': 0.0003400071674038,
    'V2_SNR_dB': 26.294086494457968,
    'V2_Q_wave_amp': 0.0858056847448529,
    'V3_STFT_energy': 0.0003321840446044,
    'V3_SNR_dB': 25.54709599177766,
    'V3_Q_wave_amp': 0.0991808385300087,
    'V4_STFT_energy': 0.0003117744458376,
    'V4_SNR_dB': 24.69671032868033,
    'V5_SNR_dB': 24.06909097559748,
    'V5_prop_censored': 0.012,
    'V5_Q_wave_amp': 0.094644480207957,
    'V6_STFT_energy': 0.0001838907542697,
    'V6_SNR_dB': 22.716601543078664,
    'V6_prop_censored': 0.012,
    'V3_PSD_mean_freq': 8.84637923148137,
    'V5_PSD_std_freq': 7.565503042060861,
    'AVF_wavelet_E5': 0.43049308099902,
    'I_wavelet_E5': 0.8095093677344505,
    'I_wavelet_entropy': 5.603074385647278
}

CATEGORICAL_COLUMNS = ['heart_axis_norm', 'V1_pathological_Q']
NUMERIC_COLUMNS = [
    'age', 'I_prop_censored', 'II_wavelet_entropy', 'II_SNR_dB', 'III_PSD_std_freq', 
    'III_wavelet_E2', 'AVR_RMSSD', 'AVR_STFT_energy', 'AVR_SNR_dB', 'AVR_prop_censored', 
    'AVL_STFT_energy', 'AVL_SNR_dB', 'AVF_SNR_dB', 'V1_RMSSD', 'V1_STFT_energy', 
    'V1_wavelet_entropy', 'V1_SNR_dB', 'V2_STFT_energy', 'V2_SNR_dB', 'V2_Q_wave_amp', 
    'V3_STFT_energy', 'V3_SNR_dB', 'V3_Q_wave_amp', 'V4_STFT_energy', 'V4_SNR_dB', 
    'V5_SNR_dB', 'V5_prop_censored', 'V5_Q_wave_amp', 'V6_STFT_energy', 'V6_SNR_dB', 
    'V6_prop_censored', 'meanRR_global', 'V3_PSD_mean_freq', 'V5_PSD_std_freq', 
    'AVF_wavelet_E5', 'I_wavelet_E5', 'I_wavelet_entropy'
]


def preprocess_features_for_model(features_dict, patient_meta, scaler, trained_columns_order):
    """
    Полная предобработка признаков как в ноутбуке:
    1. Создание DataFrame с признаками и метаданными
    2. Категоризация heart_axis
    3. Создание meanRR_global
    4. Обработка нулей и NaN
    5. Заполнение медианами
    6. Масштабирование
    7. Объединение категориальных признаков
    """
    # 1. Создаем DataFrame
    df = pd.DataFrame([features_dict])
    df['age'] = patient_meta['age']
    df['sex'] = patient_meta.get('sex', 'M')
    heart_axis = patient_meta.get('heart_axis', 'normal')
    if pd.isna(heart_axis) or heart_axis == "":
        heart_axis = 'normal'
    df['heart_axis'] = heart_axis
    
    # 2. Категоризация heart_axis
    df['heart_axis_cat'] = df['heart_axis'].apply(categorize_heart_axis)
    df['heart_axis_norm'] = df['heart_axis_cat'].apply(lambda x: 1 if x == 'normal' else 0)
    
    # 3. Обработка V1_pathological_Q (из признаков)
    if 'V1_pathological_Q' in df.columns:
        df['V1_pathological_Q'] = df['V1_pathological_Q'].astype(int)
    else:
        # Если нет в признаках, вычисляем из Q_wave_amp
        if 'V1_Q_wave_amp' in df.columns:
            df['V1_pathological_Q'] = (df['V1_Q_wave_amp'] < -0.2).astype(int)
        else:
            df['V1_pathological_Q'] = 0
    
    # 4. Создание meanRR_global
    numeric_cols = NUMERIC_COLUMNS.copy()
    meanr_list = create_col_list_from_search_string(df, 'meanR')
    if meanr_list:
        df['meanRR_global'] = df[meanr_list].mean(axis=1)
        numeric_cols = [col for col in numeric_cols if col not in meanr_list]
    
    # 5. Добавляем отсутствующие колонки как NaN
    for column in numeric_cols:
        if column not in df.columns:
            df[column] = np.nan
    
    # 6. Замена нулей на NaN (кроме valid_zero_features)
    valid_zero_features = ['prop_censored', 'PT_ratio', 'Q_wave_amp', 'Q_wave_dur']
    columns_to_process = [col for col in numeric_cols if not any(feat in col for feat in valid_zero_features)]
    for col in columns_to_process:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # 7. Заполнение NaN медианными значениями
    for column in numeric_cols:
        if column in df.columns:
            if column in REPLACEMENT_VALUES:
                df[column] = df[column].fillna(REPLACEMENT_VALUES[column])
    
    # 8. Выбираем только нужные колонки
    all_required_columns = CATEGORICAL_COLUMNS + numeric_cols
    df = df[all_required_columns]
    df_numeric_before_scale = df[numeric_cols]
    
    # 9. Масштабирование числовых признаков
    df_numeric_ordered = df_numeric_before_scale.reindex(columns=list(scaler.feature_names_in_), fill_value=0.0)
    df_numeric_scaled = pd.DataFrame(
        scaler.transform(df_numeric_ordered),
        columns=list(scaler.feature_names_in_)
    )
    
    # 10. Объединение категориальных и числовых
    df_cat = df[CATEGORICAL_COLUMNS]
    df_final = pd.concat([df_numeric_scaled, df_cat], axis=1)
    df_final = df_final.reindex(columns=trained_columns_order, fill_value=0.0)
    
    return df_final


def extract_features_for_model(ecg_record, fs, patient_meta, scaler, trained_columns_order):
    """
    Полная функция извлечения и предобработки признаков как в ноутбуке
    """
    # Извлечение признаков
    features = extract_features_from_ecg_record(ecg_record, fs)
    
    # Предобработка
    df_final = preprocess_features_for_model(features, patient_meta, scaler, trained_columns_order)
    
    return df_final

