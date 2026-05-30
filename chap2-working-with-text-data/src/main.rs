use std::path::Path;
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;
    let input1 =
        "Hello, do you like tea? <|endoftext|> In the sunlight terraces of someunknownPlace.";
    println!("Input: {}", input1);
    let output1 = tokenizer.encode(input1, false)?;
    println!("Tokens: {:?}", output1.get_tokens());
    println!("Ids: {:?}", output1.get_ids());
    let decode = tokenizer.decode(output1.get_ids(), false)?;
    println!("Decoded: {:?}\n", decode);

    let input2 = "Akwirw ier";
    println!("Input: {}", input2);
    let output2 = tokenizer.encode(input2, false)?;
    println!("Tokens: {:?}", output2.get_tokens());
    println!("Ids: {:?}", output2.get_ids());

    let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/the-verdict.txt");
    let input3 = tokio::fs::read_to_string(file_path).await?;
    println!("Input length: {:?}", input3.split_whitespace().count());

    let output3 = tokenizer.encode(input3, false)?;
    println!("Tokens length: {:?}", output3.get_tokens().len());

    let context_size = 4;
    let int_out = &output3.get_ids()[50..];
    for i in 0..context_size {
        let context = tokenizer.decode(&int_out[..i+1], false)?;
        let desired = tokenizer.decode(&[int_out[i+1]], false)?;
        println!("{context} ----> {desired}");
    }
    
    Ok(())
}
