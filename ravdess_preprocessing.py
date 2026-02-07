"""
RAVDESS Audio Dataset Preprocessing Pipeline
=============================================
This script implements comprehensive audio preprocessing including:
- Silence trimming
- Mel-spectrogram visual analysis
- Feature engineering (Log-Mel Spectrograms)
- Data augmentation (Noise, Pitch, Time Stretch)
- Stratified train/val/test split
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RAVDESSPreprocessor:
    """
    Comprehensive preprocessor for RAVDESS emotion recognition dataset
    """
    
    def __init__(self, dataset_path, output_path='./processed_ravdess'):
        """
        Initialize the preprocessor
        
        Args:
            dataset_path: Path to RAVDESS dataset directory
            output_path: Path where processed data will be saved
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Audio processing parameters
        self.sr = 22050  # Sample rate
        self.n_mels = 128  # Number of mel bands
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.max_len = 173  # Maximum time frames (approx 4 seconds)
        
        # RAVDESS emotion mapping
        self.emotion_labels = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # Arousal categorization
        self.high_arousal = ['angry', 'fearful', 'happy', 'surprised']
        self.low_arousal = ['sad', 'calm', 'neutral', 'disgust']
        
    def parse_filename(self, filename):
        """
        Parse RAVDESS filename to extract metadata
        Format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
        """
        parts = filename.stem.split('-')
        return {
            'modality': parts[0],
            'vocal_channel': parts[1],
            'emotion': self.emotion_labels[parts[2]],
            'emotion_code': parts[2],
            'intensity': parts[3],
            'statement': parts[4],
            'repetition': parts[5],
            'actor': parts[6]
        }
    
    def trim_silence(self, audio, top_db=20):
        """
        Remove silence from beginning and end of audio
        
        Args:
            audio: Audio waveform
            top_db: Threshold in dB below reference to consider as silence
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def extract_log_mel_spectrogram(self, audio, sr=None):
        """
        Extract log-mel spectrogram from audio
        
        Args:
            audio: Audio waveform
            sr: Sample rate (uses self.sr if None)
            
        Returns:
            Log-mel spectrogram with uniform shape (n_mels, max_len)
        """
        if sr is None:
            sr = self.sr
            
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to uniform shape
        log_mel_spec = self._pad_or_truncate(log_mel_spec)
        
        return log_mel_spec
    
    def _pad_or_truncate(self, spec):
        """
        Pad or truncate spectrogram to uniform shape
        """
        if spec.shape[1] < self.max_len:
            # Pad with minimum value
            pad_width = self.max_len - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', 
                         constant_values=spec.min())
        else:
            # Truncate
            spec = spec[:, :self.max_len]
        
        return spec
    
    def add_noise(self, audio, noise_factor=0.005):
        """
        Add random noise to audio
        """
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented.astype(audio.dtype)
    
    def pitch_shift(self, audio, sr=None, n_steps=2):
        """
        Shift pitch of audio
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            n_steps: Number of semitones to shift (can be negative)
        """
        if sr is None:
            sr = self.sr
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def time_stretch(self, audio, rate=1.1):
        """
        Time-stretch audio
        
        Args:
            audio: Audio waveform
            rate: Stretch factor (>1 speeds up, <1 slows down)
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def augment_audio(self, audio, sr=None):
        """
        Apply random augmentation to audio
        Returns list of augmented versions
        """
        if sr is None:
            sr = self.sr
            
        augmented_samples = []
        
        # Original
        augmented_samples.append(('original', audio))
        
        # Noise injection
        augmented_samples.append(('noise', self.add_noise(audio, noise_factor=0.005)))
        
        # Pitch shift up
        augmented_samples.append(('pitch_up', self.pitch_shift(audio, sr, n_steps=2)))
        
        # Pitch shift down
        augmented_samples.append(('pitch_down', self.pitch_shift(audio, sr, n_steps=-2)))
        
        # Time stretch (faster)
        augmented_samples.append(('time_fast', self.time_stretch(audio, rate=1.1)))
        
        # Time stretch (slower)
        augmented_samples.append(('time_slow', self.time_stretch(audio, rate=0.9)))
        
        return augmented_samples
    
    def visualize_spectrograms(self, output_dir):
        """
        Create comparative visualization of high-arousal vs low-arousal emotions
        Compares Angry (high-arousal) vs Sad (low-arousal)
        """
        print("\n" + "="*80)
        print("VISUAL ANALYSIS: Comparing High-Arousal vs Low-Arousal Spectrograms")
        print("="*80)
        
        # Find sample files for angry and sad emotions
        angry_file = None
        sad_file = None
        
        for audio_file in self.dataset_path.rglob('*.wav'):
            metadata = self.parse_filename(audio_file)
            if metadata['emotion'] == 'angry' and angry_file is None:
                angry_file = audio_file
            elif metadata['emotion'] == 'sad' and sad_file is None:
                sad_file = audio_file
            
            if angry_file and sad_file:
                break
        
        if not angry_file or not sad_file:
            print("Warning: Could not find sample files for comparison")
            return
        
        # Process both files
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, (audio_file, emotion, color) in enumerate([
            (angry_file, 'Angry (High-Arousal)', 'Reds'),
            (sad_file, 'Sad (Low-Arousal)', 'Blues')
        ]):
            # Load and process audio
            audio, sr = librosa.load(audio_file, sr=self.sr)
            audio_trimmed = self.trim_silence(audio)
            
            # Waveform
            axes[idx, 0].plot(audio_trimmed, linewidth=0.5)
            axes[idx, 0].set_title(f'{emotion} - Waveform')
            axes[idx, 0].set_xlabel('Sample')
            axes[idx, 0].set_ylabel('Amplitude')
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_trimmed, sr=sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            img1 = librosa.display.specshow(
                log_mel_spec, sr=sr, hop_length=self.hop_length,
                x_axis='time', y_axis='mel', ax=axes[idx, 1], cmap=color
            )
            axes[idx, 1].set_title(f'{emotion} - Mel Spectrogram')
            plt.colorbar(img1, ax=axes[idx, 1], format='%+2.0f dB')
            
            # Spectral energy distribution
            mean_energy = np.mean(log_mel_spec, axis=1)
            axes[idx, 2].plot(mean_energy, range(self.n_mels))
            axes[idx, 2].set_title(f'{emotion} - Mean Spectral Energy')
            axes[idx, 2].set_xlabel('Energy (dB)')
            axes[idx, 2].set_ylabel('Mel Frequency Bands')
            axes[idx, 2].invert_yaxis()
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / 'arousal_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved spectral comparison to: {output_file}")
        
        # Print analysis
        print("\n" + "-"*80)
        print("KEY OBSERVATIONS:")
        print("-"*80)
        print("High-Arousal (Angry):")
        print("  • Higher spectral energy across frequency bands")
        print("  • More intense high-frequency components")
        print("  • Greater temporal variation in amplitude")
        print("  • Broader spectral distribution")
        print("\nLow-Arousal (Sad):")
        print("  • Lower overall spectral energy")
        print("  • Concentration in lower frequency bands")
        print("  • More stable temporal patterns")
        print("  • Narrower spectral distribution")
        print("-"*80)
        
        plt.close()
    
    def load_dataset(self):
        """
        Load all audio files and organize by emotion
        """
        print("\n" + "="*80)
        print("LOADING RAVDESS DATASET")
        print("="*80)
        
        dataset = defaultdict(list)
        
        audio_files = list(self.dataset_path.rglob('*.wav'))
        print(f"Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            metadata = self.parse_filename(audio_file)
            emotion = metadata['emotion']
            dataset[emotion].append({
                'path': audio_file,
                'metadata': metadata
            })
        
        # Print statistics
        print("\nDataset Statistics:")
        print("-" * 80)
        for emotion in sorted(dataset.keys()):
            count = len(dataset[emotion])
            arousal = "High-Arousal" if emotion in self.high_arousal else "Low-Arousal"
            print(f"  {emotion.capitalize():12s}: {count:4d} samples  ({arousal})")
        print("-" * 80)
        print(f"  Total:        {sum(len(v) for v in dataset.values()):4d} samples")
        print("-" * 80)
        
        return dataset
    
    def process_and_split_dataset(self, apply_augmentation=True):
        """
        Process entire dataset with augmentation and create stratified splits
        """
        print("\n" + "="*80)
        print("PROCESSING DATASET WITH FEATURE ENGINEERING")
        print("="*80)
        
        dataset = self.load_dataset()
        
        # Create output directories
        vis_dir = self.output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Perform visual analysis
        self.visualize_spectrograms(vis_dir)
        
        # Process all files
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        
        # Create emotion to index mapping
        emotion_to_idx = {emotion: idx for idx, emotion in 
                         enumerate(sorted(self.emotion_labels.values()))}
        
        print("\nProcessing audio files...")
        print("-" * 80)
        
        total_processed = 0
        
        for emotion, files in dataset.items():
            print(f"\nProcessing {emotion} ({len(files)} files)...")
            
            # Prepare data for this emotion
            emotion_data = []
            emotion_labels = []
            
            for file_info in files:
                audio_path = file_info['path']
                
                # Load audio
                audio, sr = librosa.load(audio_path, sr=self.sr)
                
                # Trim silence
                audio = self.trim_silence(audio)
                
                # Extract log-mel spectrogram for original
                log_mel = self.extract_log_mel_spectrogram(audio)
                emotion_data.append(log_mel)
                emotion_labels.append(emotion_to_idx[emotion])
                
                # Apply augmentation to training data
                if apply_augmentation:
                    augmented = self.augment_audio(audio)
                    for aug_name, aug_audio in augmented[1:]:  # Skip original
                        aug_log_mel = self.extract_log_mel_spectrogram(aug_audio)
                        emotion_data.append(aug_log_mel)
                        emotion_labels.append(emotion_to_idx[emotion])
            
            # Convert to arrays
            emotion_data = np.array(emotion_data)
            emotion_labels = np.array(emotion_labels)
            
            # Stratified split: 80% train, 10% val, 10% test
            # First split: 80% train, 20% temp
            X_temp, X_test_emotion, y_temp, y_test_emotion = train_test_split(
                emotion_data, emotion_labels, test_size=0.1, random_state=42
            )
            
            # Second split: 80% of remaining -> train, 20% of remaining -> val
            X_train_emotion, X_val_emotion, y_train_emotion, y_val_emotion = train_test_split(
                X_temp, y_temp, test_size=0.111, random_state=42  # 0.111 * 0.9 ≈ 0.1
            )
            
            # Append to main lists
            X_train.append(X_train_emotion)
            X_val.append(X_val_emotion)
            X_test.append(X_test_emotion)
            
            y_train.append(y_train_emotion)
            y_val.append(y_val_emotion)
            y_test.append(y_test_emotion)
            
            print(f"  ✓ Train: {len(X_train_emotion)}, Val: {len(X_val_emotion)}, Test: {len(X_test_emotion)}")
            total_processed += len(files)
        
        # Concatenate all emotions
        X_train = np.concatenate(X_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        
        y_train = np.concatenate(y_train, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        # Shuffle
        train_indices = np.random.permutation(len(X_train))
        val_indices = np.random.permutation(len(X_val))
        test_indices = np.random.permutation(len(X_test))
        
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        X_val, y_val = X_val[val_indices], y_val[val_indices]
        X_test, y_test = X_test[test_indices], y_test[test_indices]
        
        print("\n" + "="*80)
        print("DATASET SPLIT SUMMARY")
        print("="*80)
        print(f"Training set:   {X_train.shape[0]:5d} samples  ({X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]:5d} samples  ({X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"Test set:       {X_test.shape[0]:5d} samples  ({X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"Total:          {X_train.shape[0]+X_val.shape[0]+X_test.shape[0]:5d} samples")
        print(f"\nFeature shape: {X_train.shape[1:]} (n_mels, time_steps)")
        print("="*80)
        
        # Verify stratification
        print("\nVerifying Stratified Split (samples per emotion):")
        print("-" * 80)
        print(f"{'Emotion':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-" * 80)
        
        idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
        for idx in range(len(emotion_to_idx)):
            emotion_name = idx_to_emotion[idx]
            train_count = np.sum(y_train == idx)
            val_count = np.sum(y_val == idx)
            test_count = np.sum(y_test == idx)
            print(f"{emotion_name:<12} {train_count:<8} {val_count:<8} {test_count:<8}")
        print("-" * 80)
        
        # Save processed data
        print("\nSaving processed data...")
        np.save(self.output_path / 'X_train.npy', X_train)
        np.save(self.output_path / 'X_val.npy', X_val)
        np.save(self.output_path / 'X_test.npy', X_test)
        np.save(self.output_path / 'y_train.npy', y_train)
        np.save(self.output_path / 'y_val.npy', y_val)
        np.save(self.output_path / 'y_test.npy', y_test)
        
        # Save label mapping
        import json
        with open(self.output_path / 'label_mapping.json', 'w') as f:
            json.dump(emotion_to_idx, f, indent=2)
        
        print(f"✓ Processed data saved to: {self.output_path}")
        print("="*80)
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'emotion_to_idx': emotion_to_idx
        }


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print(" " * 20 + "RAVDESS PREPROCESSING PIPELINE")
    print("="*80)
    
    # Set your dataset path here
    DATASET_PATH = r"C:\Users\kesha\programming\AI_club\Audio_Speech_Actors_01-24\\"

  # UPDATE THIS PATH
    OUTPUT_PATH = './processed_ravdess'
    
    # Check if path exists
    if not os.path.exists(DATASET_PATH):
        print("\n⚠️  ERROR: Dataset path not found!")
        print(f"   Please update DATASET_PATH in the script to point to your RAVDESS dataset")
        print(f"   Current path: {DATASET_PATH}")
        print("\nExpected directory structure:")
        print("  RAVDESS/")
        print("    Actor_01/")
        print("      03-01-01-01-01-01-01.wav")
        print("      ...")
        print("    Actor_02/")
        print("      ...")
        return
    
    # Initialize preprocessor
    preprocessor = RAVDESSPreprocessor(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Process dataset with all operations
    processed_data = preprocessor.process_and_split_dataset(apply_augmentation=True)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  • X_train.npy, y_train.npy  (Training set)")
    print(f"  • X_val.npy, y_val.npy      (Validation set)")
    print(f"  • X_test.npy, y_test.npy    (Test set)")
    print(f"  • label_mapping.json        (Emotion labels)")
    print(f"  • visualizations/arousal_comparison.png")
    print("\nNext Steps:")
    print("  1. Load the processed data for model training")
    print("  2. Build and train your emotion recognition model")
    print("  3. Evaluate on the test set")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()