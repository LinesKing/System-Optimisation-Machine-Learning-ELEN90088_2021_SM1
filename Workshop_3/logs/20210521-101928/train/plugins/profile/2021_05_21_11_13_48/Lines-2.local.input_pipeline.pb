	?? ?rh???? ?rh??!?? ?rh??	??????!@??????!@!??????!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?? ?rh??sh??|???A;?O??n??Y?~j?t???*	     @Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/?$???!      D@)?l??????1?y??y?A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???S㥛?!m۶m۶9@){?G?z??1?0?03@:Preprocessing2U
Iterator::Model::ParallelMapV2?? ?rh??!?0?00@)?? ?rh??1?0?00@:Preprocessing2F
Iterator::Model????????!?<??<?7@)????Mb??1??y??y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{?G?z??!?0?0S@)????Mb??1??y??y@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??????@)y?&1?|?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!?0?0@){?G?zt?1?0?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!$I?$I?<@)?~j?t?h?1?m۶m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??????!@IKKKKK?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sh??|???sh??|???!sh??|???      ??!       "      ??!       *      ??!       2	;?O??n??;?O??n??!;?O??n??:      ??!       B      ??!       J	?~j?t????~j?t???!?~j?t???R      ??!       Z	?~j?t????~j?t???!?~j?t???b      ??!       JCPU_ONLYY??????!@b qKKKKK?V@