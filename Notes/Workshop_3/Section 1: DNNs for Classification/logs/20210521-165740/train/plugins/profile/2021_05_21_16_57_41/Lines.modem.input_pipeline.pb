	??ʡE?????ʡE???!??ʡE???	{?f?FA@{?f?FA@!{?f?FA@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ʡE???ˡE?????A?????K??Y?z?G???*	     ?|@2U
Iterator::Model::ParallelMapV2?Zd;??!??9T,hJ@)?Zd;??1??9T,hJ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^?I+??!?Z"???C@)^?I+??1?Z"???C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!???D?o@)????Mb??1??n??@:Preprocessing2F
Iterator::ModelP??n???!5?wL?K@)y?&1???1>???>@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!?S{???)?~j?t?x?1?S{???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!??n????)????MbP?1??n????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s9.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9|?f?FA@IB?̆?{P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ˡE?????ˡE?????!ˡE?????      ??!       "      ??!       *      ??!       2	?????K???????K??!?????K??:      ??!       B      ??!       J	?z?G????z?G???!?z?G???R      ??!       Z	?z?G????z?G???!?z?G???b      ??!       JCPU_ONLYY|?f?FA@b qB?̆?{P@