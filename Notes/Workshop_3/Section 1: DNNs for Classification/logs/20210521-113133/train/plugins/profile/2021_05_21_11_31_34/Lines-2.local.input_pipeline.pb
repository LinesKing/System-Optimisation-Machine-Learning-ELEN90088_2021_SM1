	???K7??????K7???!???K7???	O??0?/@O??0?/@!O??0?/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???K7???/?$???A????x???Y?A`??"??*	     @c@2U
Iterator::Model::ParallelMapV2???x?&??!S{???E@)???x?&??1S{???E@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;?O???!{????=@){?G?z??1f?'?Y?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!P????0@)9??v????1P????0@:Preprocessing2F
Iterator::Model?MbX9??!???;?I@)?~j?t???1????8+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!????8+@)?~j?t?x?1????8+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+???!?E]t??@)?~j?t?h?1????8+??:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9O??0?/@I???U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/?$???/?$???!/?$???      ??!       "      ??!       *      ??!       2	????x???????x???!????x???:      ??!       B      ??!       J	?A`??"???A`??"??!?A`??"??R      ??!       Z	?A`??"???A`??"??!?A`??"??b      ??!       JCPU_ONLYYO??0?/@b q???U@