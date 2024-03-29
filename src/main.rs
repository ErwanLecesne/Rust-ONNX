
#[macro_use]
extern crate ndarray;
use ndarray::{Array,Array3, Axis};
use std::time::SystemTime;
use nifti::{NiftiObject, ReaderOptions, writer,IntoNdArray};
//use tract_ndarray::Array;
use tract_onnx::prelude::*;
use std::io;
use std::path::Path;
use ferris_says::*;
use rayon::prelude::*;
use std::ffi::OsStr;

//#[macro_use(s)]


fn main() {
    let start = SystemTime::now();
    let args: Vec<String> = std::env::args().collect();

    let  image_path = Path::new(&args[2]);
    let mut ext: Option<&str> = Some("nii");
    let mut input_ext: Option<&str> = image_path.extension().and_then(OsStr::to_str);
    assert_eq!(ext,input_ext);

    let  onnx_path = Path::new(&args[1]);
    ext = Some("onnx");
    input_ext = onnx_path.extension().and_then(OsStr::to_str);
    assert_eq!(ext,input_ext);
 
   

    let output_path = Path::new(&args[2]).parent().unwrap().join("output.nii");
    let mut niiwriter = writer::WriterOptions::new(&output_path);
    println!("OutPut dir : {}\n",output_path.display());

    let reader = ReaderOptions::new();

    let image = reader.read_file(&image_path).unwrap();
    //use obj
    let _header = image.header();
    let mut volume = image.into_volume().into_ndarray::<f32>().unwrap();
    let input_shape = volume.shape().to_owned();
    
    for v in input_shape.iter() {
        println!("input shape {:?}",v);
    
    }
   
    
    volume.swap_axes(1, 3);
    volume.swap_axes(0, 2); //(B,C,H,W)
   
    let mean = volume.sum() / volume.len() as f32;
    let std = volume.std(0.);

    volume = (volume - mean) / std; //Znormalization

    let dim_volume = volume.shape();


    for v in dim_volume.iter() {
        println!("Volume {:?}",v);
    
    }

      
    let model = tract_onnx::onnx()
    // load the model
    .model_for_path(onnx_path).unwrap()
    // specify input type and shape
    .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, input_shape[0], input_shape[1]))).unwrap()
    // optimize the model
    .into_optimized().unwrap()
    // make the model runnable and fix its inputs and outputs
    .into_runnable().unwrap();
    
    let mut vecslices = Vec::new();
   
    for slice in volume.axis_iter(Axis(0)){ 

        vecslices.push(tvec!(slice.to_owned()
        .insert_axis(Axis(0))
        .into()));
    }
    
    let pred_vec: Vec<_> = vecslices
    // Make the inference for each slices multithreaded
    .par_iter() 
    .filter_map(|x| (model.run(x.to_owned()).ok()))
    .collect();


    let mut prediction = Array3::<i64>::zeros((input_shape[2],input_shape[0],input_shape[1])); 
    let mut i :u8 = 0;
    
    for slice in pred_vec.iter(){   
        
        let array = (*slice[0]).clone().into_array::<i64>().unwrap() ; 
        prediction.slice_mut(s![usize::from(i),.., ..]).assign(&array.slice(s![0,.., ..]));
        i += 1;
    }
 
    let mut casted = prediction.map(|&e| e as f32);//so i have to cast it in f32


    casted.swap_axes(0, 2);
    casted.swap_axes(0, 1);
 
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