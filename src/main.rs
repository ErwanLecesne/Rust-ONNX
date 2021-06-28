extern crate ndarray;
use ndarray::{Array,Array3,Array4, Axis};
use std::time::SystemTime;
use nifti::{NiftiObject, ReaderOptions, writer, InMemNiftiObject, NiftiVolume,IntoNdArray};
//use tract_ndarray::Array;
use tract_onnx::prelude::*;
use std::io;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use ferris_says::*;

fn main() {


    
    let mut str_buf = String::new();
    let current_dir = env::current_dir().unwrap();
    println!("Your current dir is : {}\n",current_dir.display());

    println!("Enter the file path.(ex: /folder1/folder2/myfile.nii)\n ");
    io::stdin().read_line(&mut str_buf).unwrap(); 
    let image_str = str_buf.trim().to_owned();
    let image_path = Path::new(&image_str);
   
    let output_path = Path::new(&image_str).parent().unwrap().join("output.nii");
    let mut niiwriter = writer::WriterOptions::new(&output_path);
    println!("OutPut dir : {}\n",output_path.display());

    
    println!("Enter the ONNX path. ex: /folder1/folder2/myfile.onnx)\n ");

    str_buf = String::new();
    io::stdin().read_line(&mut str_buf).unwrap();
    let onnx_str = str_buf.trim().to_owned();
    let onnx_path = Path::new(&onnx_str);

    let reader = ReaderOptions::new();

    let image = reader.read_file(&image_path).unwrap();
    //use obj
    let _header = image.header();
    let mut volume = image.into_volume().into_ndarray::<f32>().unwrap();
    let _dims = volume.shape();
    /*
    for v in dims.iter() {
        println!("input shape {:?}",v);
    
    }
   */
    
    let start = SystemTime::now();
    volume.swap_axes(0, 3); 
    volume.swap_axes(1, 2); 
    volume.swap_axes(0, 1); //(B,C,H,W)
    //volume = volume.insert_axis(Axis(0));
    let mean = volume.sum() / volume.len() as f32;
    let std = volume.std(0.);

    volume = (volume - mean) / std; //Znormalization


    let model = tract_onnx::onnx()
    // load the model
    .model_for_path(onnx_path).unwrap()
    // specify input type and shape
    .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(16, 1, 128, 128))).unwrap()
    // optimize the model
    .into_optimized().unwrap()
    // make the model runnable and fix its inputs and outputs
    .into_runnable().unwrap();
    

    //to make a test => let image = Array4::<f32>::ones((16, 1, 128, 128));
    let image: Tensor = volume.into(); //

    let result = model.run(tvec!(image)).unwrap(); // make the inference 
    
    let prediction = (*result[0]).clone().into_array::<i64>().unwrap();// should modify that Pytorch return always f32
    let mut casted = prediction.map(|&e| e as f32);//so i have to cast it in f32


    let _dims_res = prediction.shape();
/*
    
    for v in dims_res.iter() {
        println!("output shape {:?}",v);
    }
*/

    casted.swap_axes(0, 2);

    

    let _dims_pred = casted.shape();
/*
    for v in dims_pred.iter() {
        println!("prediction shape  {:?}",v);
    }
   */ 
    niiwriter = writer::WriterOptions::new(output_path);
    let _wrote = niiwriter.write_nifti(&casted);

    let end = SystemTime::now();
    let difference = end.duration_since(start);
    println!("time to make it {:?}",difference);  
   

    let stdout = io::stdout();
    let out = b"Inference done !";
    let width = 24;

    let mut writer = io::BufWriter::new(stdout.lock());
    say(out, width, &mut writer).unwrap();
   
}