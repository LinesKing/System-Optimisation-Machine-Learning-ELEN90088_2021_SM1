	q=
ףp??q=
ףp??!q=
ףp??	"5?x+?0@"5?x+?0@!"5?x+?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ףp??sh??|???AX9??v??YH?z?G??*	     ?m@2U
Iterator::Model::ParallelMapV2Zd;?O???![4??}C@)Zd;?O???1[4??}C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapˡE?????!W'u_A@)ˡE?????1W'u_A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?l??????!'u_[/@)????Mb??1'u_+@:Preprocessing2F
Iterator::Model???S㥻?!??؊??F@)????Mb??1'u_@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??n??!???c+?@);?O??n??1???c+?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!"h8??? @){?G?zt?1"h8??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 17.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9"5?x+?0@I???!5?T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sh??|???sh??|???!sh??|???      ??!       "      ??!       *      ??!       2	X9??v??X9??v??!X9??v??:      ??!       B      ??!       J	H?z?G??H?z?G??!H?z?G??R      ??!       Z	H?z?G??H?z?G??!H?z?G??b      ??!       JCPU_ONLYY"5?x+?0@b q???!5?T@