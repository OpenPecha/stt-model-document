import time
from transformers import WhisperProcessor

def measure_tokenization(tokenizer, text, name=""):
    # Measure tokenization time
    start_time = time.time()
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    end_time = time.time()
    
    tokenization_time = (end_time - start_time) * 1000  # Convert to milliseconds
    input_ids = encoding["input_ids"]
    sequence_length = input_ids.shape[1]
    vocab_size = len(tokenizer.get_vocab())
    
    print(f"\n{name} Tokenizer Statistics:")
    print(f"Tokenization Time: {tokenization_time:.2f} ms")
    print(f"Sequence Length: {sequence_length}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Input IDs: {input_ids}")
    
    return tokenization_time, sequence_length, vocab_size

def main():
    # Test text
    test_text = "ཁོས་གཞན་ལ་ཁོང་ཁྲོ་སླངས་ན། དེ་ནས་ཁོལ་ཚང་མས་ཐལ་མོ་བརྡབ་ཀྱི་རེད་བ། དཔེ་དེ་ལྟ་བུ། གཞན་གྱི་ཡོན་ཏན་བརྟགས་ན། དེ་ལ་དེ་ནས་བརྩེ་བ་སྐྱེ་གི་རེད་བ། དཔེར་ན་གྲོགས་གཅིག་ཡོད་ན། ཨོ་ང་ལ་གྲོགས་དེས་འདི་འདྲའི་བསམ་གྱི་ཡོད། ང་ལ་བརྩེ་བ་ཡོད། ད་ཁོང་ལས་ཀ་བཟང་པོ་དེ་འདྲས་ཡོན་ཏན་ལ་བརྟག་ན། ཡོན་ཏན་ཀྱི་ཆ་ཐམས་ཅད་ནས་བལྟས་ན། མི་དེ་ཡོན་ཏན་གྱི་ཕུང་པོ་ནས་མཐོང་གི་ཡོད་རེད། དེ་ལ་བརྩེ་བ་སྐྱེ་གི་རེད། སྐྱོན་ལ་རྟོགས་ན་དེ་ནས་སྐྱོན་གྱི་ཕུང་པོ་ལ་མཐོང་གི་ཡོད་རེད། དེ་ན། ང་ཚོས་བྱམས་སྙིང་རྗེ་སྐྱེས་བའི་ཐབས་ལ་གཞན་གྱི་ཡོན་ཏན་བརྟགས་དགོས་རེད་ཟེར། བརྩེ་བ་དང་གུས་པ་སྐྱེས་པའི་དོན་ལ་"
    test_wylie_text = "khos gzhan la khong khro slangs na/_de nas khol tshang mas thal mo brdab kyi red ba/_dpe de lta bu/_gzhan gyi yon tan brtags na/_de la de nas brtse ba skye gi red ba/_dper na grogs gcig yod na/_o nga la grogs des 'di 'dra'i bsam gyi yod/_nga la brtse ba yod/_da khong las ka bzang po de 'dras yon tan la brtag na/_yon tan kyi cha thams cad nas bltas na/_mi de yon tan gyi phung po nas mthong gi yod red/_de la brtse ba skye gi red/_skyon la rtogs na de nas skyon gyi phung po la mthong gi yod red/_de na/_nga tshos byams snying rje skyes ba'i thabs la gzhan gyi yon tan brtags dgos red zer/_brtse ba dang gus pa skyes pa'i don la"
    # Load default Whisper tokenizer
    default_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # Load your custom tokenizer with added Tibetan tokens
    custom_processor = WhisperProcessor.from_pretrained("ganga4364/whisper-small-latin-added-tibetan-checkpoint-4000")
    
    
    
    # Measure custom tokenizer
    custom_time, custom_length, custom_vocab = measure_tokenization(
        custom_processor.tokenizer, 
        test_text, 
        "Custom Tibetan"
    )
    
    # Measure default tokenizer
    default_time, default_length, default_vocab = measure_tokenization(
        default_processor.tokenizer, 
        test_wylie_text, 
        "Default Whisper"
    )
    # Calculate and print ratios
    print("\nComparison Ratios (Custom/Default):")
    print(f"Time Ratio: {custom_time/default_time:.2f}x")
    print(f"Length Ratio: {custom_length/default_length:.2f}x")
    print(f"Vocabulary Size Ratio: {custom_vocab/default_vocab:.2f}x")
    
    # Print added tokens count
    added_tokens = custom_vocab - default_vocab
    print(f"\nNumber of Added Tokens: {added_tokens}")

if __name__ == "__main__":
    main()
