	????x???????x???!????x???	)<?_?4@)<?_?4@!)<?_?4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????x???+??????A?????K??Y?O??n??*	      l@2U
Iterator::Model::ParallelMapV2???S㥻?!?m۶mH@)???S㥻?1?m۶mH@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???x?&??!?$I?$?=@)???x?&??1?$I?$?=@:Preprocessing2F
Iterator::ModelX9??v???!ܶm۶?K@)????Mb??1$I?$I?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!$I?$I?@)????Mb??1$I?$I?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!?m۶m[@)9??v????1n۶m?6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!$I?$I???)????Mbp?1$I?$I???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 20.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t10.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9)<?_?4@I??p(?S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+??????+??????!+??????      ??!       "      ??!       *      ??!       2	?????K???????K??!?????K??:      ??!       B      ??!       J	?O??n???O??n??!?O??n??R      ??!       Z	?O??n???O??n??!?O??n??b      ??!       JCPU_ONLYY)<?_?4@b q??p(?S@