def split_range_and_call_function(start, end):
    chunk_size = 100
    while start < end:
        chunk_end = min(start + chunk_size - 1, end)
        print(f"Calling function with start={start} and end={chunk_end}")
        # Replace the print statement with the actual function call
        # your_function(start, chunk_end)
        start = chunk_end + 1

# Example usage:
#split_range_and_call_function(0, 999)  # Should split into 10 parts
split_range_and_call_function(0, 8) # Should split into 11 parts

#%%
