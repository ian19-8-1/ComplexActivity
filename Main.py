from Datasets import BreakfastTexts, BreakfastClips


bf_texts = BreakfastTexts('embeddings.pkl')
print('Shape:', len(bf_texts), 'x', len(bf_texts[0]))

bf_clips = BreakfastClips()
print('Shape:', len(bf_clips), 'x', len(bf_clips[0]))
sample = bf_clips.get_sample(6, 5)
print('Sample:', len(sample), 'x', len(sample[0]))
