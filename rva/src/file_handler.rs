use std::env;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

pub struct FileHandle{
    name   : String,
    byline : Vec<String>
}


impl FileHandle {

    pub fn default() -> Self{
        return FileHandle {
            name   : String::default(),
            byline : Vec::default()
        }
    }

    pub fn build( args : Vec<String>) -> Result<Self, &'static str> {
        if args.len() < 2 {
            return Err("No Input File");
        }

        let fname : String = args[1].to_string();
        let fpo = File::open(fname.clone());
        
        if fpo.is_err() {
            return Err("Cannot open file");
        }

        let reader = BufReader::new(fpo.unwrap());
        
        let mut lines : Vec<String> = Vec::default();

        for line in reader.lines() {
            if line.is_ok() {
                lines.push(line.unwrap())
            }
        }

        return Ok(FileHandle {
            name : fname,
            byline : lines
        })
    }
    
    pub fn get_name(&self) -> String {
        return self.name.to_string();
    }

    pub fn get_data(&self) -> &Vec<String> {
        return &self.byline;
    }

}
