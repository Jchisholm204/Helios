use std::env;
// use std::io::File;
// use std::io::prelude::*;
use std::process::exit;
pub mod file_handler;
use file_handler::FileHandle;

fn main() {
    let args: Vec<String> = env::args().collect();

    let fh : file_handler::FileHandle = match FileHandle::build(args) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("RVA Fatal Error:\n {}", e);
            exit(1);
        }
    };

    println!("Parsing File : {}", fh.get_name());

    for (i, line) in fh.get_data().into_iter().enumerate() {
        println!("{}: {}", i, line);
    }

    // println!("Parsing File: {}", fh.file)
    // let input : String = args.get(1).unwrap().to_string();
    // = args.get(1);
    // println!("Parsing File {}", input);
}
